from unittest.mock import MagicMock, patch

import pytest
from gitlab import Gitlab
from gitlab.exceptions import GitlabGetError
from gitlab.v4.objects import Project, ProjectFile

from pr_agent.algo.types import FilePatchInfo
from pr_agent.git_providers.gitlab_provider import GitLabProvider


class TestGitLabProvider:
    """Test suite for GitLab provider functionality."""

    @pytest.fixture
    def mock_gitlab_client(self):
        client = MagicMock()
        return client

    @pytest.fixture
    def mock_project(self):
        project = MagicMock()
        return project

    @pytest.fixture
    def gitlab_provider(self, mock_gitlab_client, mock_project):
        with patch('pr_agent.git_providers.gitlab_provider.gitlab.Gitlab', return_value=mock_gitlab_client), \
             patch('pr_agent.git_providers.gitlab_provider.get_settings') as mock_settings:

            mock_settings.return_value.get.side_effect = lambda key, default=None: {
                "GITLAB.URL": "https://gitlab.com",
                "GITLAB.PERSONAL_ACCESS_TOKEN": "fake_token"
            }.get(key, default)

            mock_gitlab_client.projects.get.return_value = mock_project
            provider = GitLabProvider("https://gitlab.com/test/repo/-/merge_requests/1")
            provider.gl = mock_gitlab_client
            provider.id_project = "test/repo"
            return provider

    def test_get_pr_file_content_success(self, gitlab_provider, mock_project):
        mock_file = MagicMock(ProjectFile)
        mock_file.decode.return_value = "# Changelog\n\n## v1.0.0\n- Initial release"
        mock_project.files.get.return_value = mock_file

        content = gitlab_provider.get_pr_file_content("CHANGELOG.md", "main")

        assert content == "# Changelog\n\n## v1.0.0\n- Initial release"
        mock_project.files.get.assert_called_once_with("CHANGELOG.md", "main")
        mock_file.decode.assert_called_once()

    def test_get_pr_file_content_with_bytes(self, gitlab_provider, mock_project):
        mock_file = MagicMock(ProjectFile)
        mock_file.decode.return_value = b"# Changelog\n\n## v1.0.0\n- Initial release"
        mock_project.files.get.return_value = mock_file

        content = gitlab_provider.get_pr_file_content("CHANGELOG.md", "main")

        assert content == "# Changelog\n\n## v1.0.0\n- Initial release"
        mock_project.files.get.assert_called_once_with("CHANGELOG.md", "main")

    def test_get_pr_file_content_file_not_found(self, gitlab_provider, mock_project):
        mock_project.files.get.side_effect = GitlabGetError("404 Not Found")

        content = gitlab_provider.get_pr_file_content("CHANGELOG.md", "main")

        assert content == ""
        mock_project.files.get.assert_called_once_with("CHANGELOG.md", "main")

    def test_get_pr_file_content_other_exception(self, gitlab_provider, mock_project):
        mock_project.files.get.side_effect = Exception("Network error")

        content = gitlab_provider.get_pr_file_content("CHANGELOG.md", "main")

        assert content == ""

    def test_create_or_update_pr_file_create_new(self, gitlab_provider, mock_project):
        mock_project.files.get.side_effect = GitlabGetError("404 Not Found")
        mock_file = MagicMock()
        mock_project.files.create.return_value = mock_file

        new_content = "# Changelog\n\n## v1.1.0\n- New feature"
        commit_message = "Add CHANGELOG.md"

        gitlab_provider.create_or_update_pr_file(
            "CHANGELOG.md", "feature-branch", new_content, commit_message
        )

        mock_project.files.get.assert_called_once_with("CHANGELOG.md", "feature-branch")
        mock_project.files.create.assert_called_once_with({
            'file_path': 'CHANGELOG.md',
            'branch': 'feature-branch',
            'content': new_content,
            'commit_message': commit_message,
        })

    def test_create_or_update_pr_file_update_existing(self, gitlab_provider, mock_project):
        mock_file = MagicMock(ProjectFile)
        mock_file.decode.return_value = "# Old changelog content"
        mock_project.files.get.return_value = mock_file

        new_content = "# New changelog content"
        commit_message = "Update CHANGELOG.md"

        gitlab_provider.create_or_update_pr_file(
            "CHANGELOG.md", "feature-branch", new_content, commit_message
        )

        mock_project.files.get.assert_called_once_with("CHANGELOG.md", "feature-branch")
        mock_file.content = new_content
        mock_file.save.assert_called_once_with(branch="feature-branch", commit_message=commit_message)

    def test_create_or_update_pr_file_update_exception(self, gitlab_provider, mock_project):
        mock_project.files.get.side_effect = Exception("Network error")

        with pytest.raises(Exception):
            gitlab_provider.create_or_update_pr_file(
                "CHANGELOG.md", "feature-branch", "content", "message"
            )

    def test_has_create_or_update_pr_file_method(self, gitlab_provider):
        assert hasattr(gitlab_provider, "create_or_update_pr_file")
        assert callable(getattr(gitlab_provider, "create_or_update_pr_file"))

    def test_method_signature_compatibility(self, gitlab_provider):
        import inspect

        sig = inspect.signature(gitlab_provider.create_or_update_pr_file)
        params = list(sig.parameters.keys())

        expected_params = ['file_path', 'branch', 'contents', 'message']
        assert params == expected_params

    @pytest.mark.parametrize("content,expected", [
        ("simple text", "simple text"),
        (b"bytes content", "bytes content"),
        ("", ""),
        (b"", ""),
        ("unicode: café", "unicode: café"),
        (b"unicode: caf\xc3\xa9", "unicode: café"),
    ])
    def test_content_encoding_handling(self, gitlab_provider, mock_project, content, expected):
        mock_file = MagicMock(ProjectFile)
        mock_file.decode.return_value = content
        mock_project.files.get.return_value = mock_file

        result = gitlab_provider.get_pr_file_content("test.md", "main")

        assert result == expected

    def test_get_gitmodules_map_parsing(self, gitlab_provider, mock_project):
        gitlab_provider.id_project = "1"
        gitlab_provider.mr = MagicMock()
        gitlab_provider.mr.target_branch = "main"

        file_obj = MagicMock(ProjectFile)
        file_obj.decode.return_value = (
            "[submodule \"libs/a\"]\n"
            "    path = \"libs/a\"\n"
            "    url = \"https://gitlab.com/a.git\"\n"
            "[submodule \"libs/b\"]\n"
            "    path = libs/b\n"
            "    url = git@gitlab.com:b.git\n"
        )
        mock_project.files.get.return_value = file_obj
        gitlab_provider.gl.projects.get.return_value = mock_project

        result = gitlab_provider._get_gitmodules_map()
        assert result == {
            "libs/a": "https://gitlab.com/a.git",
            "libs/b": "git@gitlab.com:b.git",
        }

    def test_project_by_path_requires_exact_match(self, gitlab_provider):
        gitlab_provider.gl.projects.get.reset_mock()
        gitlab_provider.gl.projects.get.side_effect = Exception("not found")
        fake = MagicMock()
        fake.path_with_namespace = "other/group/repo"
        gitlab_provider.gl.projects.list.return_value = [fake]

        result = gitlab_provider._project_by_path("group/repo")

        assert result is None
        assert gitlab_provider.gl.projects.get.call_count == 2

    def test_compare_submodule_cached(self, gitlab_provider):
        proj = MagicMock()
        proj.repository_compare.return_value = {"diffs": [{"diff": "d"}]}
        with patch.object(gitlab_provider, "_project_by_path", return_value=proj) as m_pbp:
            first = gitlab_provider._compare_submodule("grp/repo", "old", "new")
            second = gitlab_provider._compare_submodule("grp/repo", "old", "new")

        assert first == second == [{"diff": "d"}]
        m_pbp.assert_called_once_with("grp/repo")
        proj.repository_compare.assert_called_once_with("old", "new")

    def test_send_inline_comment_logs_context_when_discussion_create_fails(self, gitlab_provider):
        gitlab_provider.id_mr = 1
        gitlab_provider.mr = MagicMock()
        gitlab_provider.mr.discussions.create.side_effect = Exception("invalid position")

        diff = MagicMock(
            base_commit_sha="base-sha",
            start_commit_sha="start-sha",
            head_commit_sha="head-sha",
        )
        target_file = FilePatchInfo(
            base_file="old line",
            head_file="new line",
            patch="@@ -1,1 +1,1 @@\n-old line\n+new line",
            filename="src/app.py",
        )
        original_suggestion = {
            "relevant_lines_start": 1,
            "relevant_lines_end": 1,
            "existing_code": "old line",
            "improved_code": "new line",
            "suggestion_content": "Use the new line",
            "label": "bug fix",
            "score": 8,
        }

        with patch.object(gitlab_provider, "get_relevant_diff", return_value=diff), \
             patch.object(gitlab_provider, "get_line_link", return_value="https://gitlab.example/src/app.py#L1"), \
             patch("pr_agent.git_providers.gitlab_provider.get_logger") as mock_get_logger:
            logger = mock_get_logger.return_value

            gitlab_provider.send_inline_comment(
                body="**Suggestion:** Use the new line\n```suggestion:-0+0\nnew line\n```",
                edit_type="addition",
                found=True,
                relevant_file="src/app.py",
                relevant_line_in_file="new line",
                source_line_no=-1,
                target_file=target_file,
                target_line_no=2,
                original_suggestion=original_suggestion,
            )

        logger.warning.assert_called_once()
        log_message = logger.warning.call_args.args[0]
        log_artifact = logger.warning.call_args.kwargs["artifact"]

        assert "Failed to create GitLab inline discussion" in log_message
        assert log_artifact["error"] == "invalid position"
        assert log_artifact["position"]["new_line"] == 1
        assert log_artifact["relevant_file"] == "src/app.py"
        assert log_artifact["relevant_lines_start"] == 1
        assert log_artifact["relevant_lines_end"] == 1

    def test_publish_code_suggestions_skips_gitlab_discussion_for_non_diff_line(self, gitlab_provider):
        gitlab_provider.mr = MagicMock()
        gitlab_provider.mr.diff_refs = {
            "base_sha": "base-ref",
            "start_sha": "start-ref",
            "head_sha": "head-ref",
        }
        gitlab_provider.diff_files = [
            FilePatchInfo(
                base_file="line 1\nline 2\nline 3",
                head_file="line 1\nline 2\nline 3",
                patch="@@ -1,3 +1,3 @@\n line 1\n-line old\n+line new\n line 3",
                filename="src/app.py",
            )
        ]
        suggestion = {
            "body": "**Suggestion:** Change line 3\n```suggestion\nline 3 changed\n```",
            "relevant_file": "src/app.py",
            "relevant_lines_start": 3,
            "relevant_lines_end": 3,
            "original_suggestion": {
                "relevant_lines_start": 3,
                "relevant_lines_end": 3,
                "existing_code": "line 3",
                "improved_code": "line 3 changed",
                "suggestion_content": "Change line 3",
                "label": "possible issue",
                "score": 7,
            },
        }

        with patch.object(gitlab_provider, "get_diff_files", return_value=gitlab_provider.diff_files), \
             patch.object(gitlab_provider, "get_line_link", return_value="https://gitlab.example/src/app.py#L3"), \
             patch("pr_agent.git_providers.gitlab_provider.get_logger"):
            gitlab_provider.publish_code_suggestions([suggestion])

        gitlab_provider.mr.discussions.create.assert_not_called()
        gitlab_provider.mr.notes.create.assert_called_once()

    def test_publish_code_suggestions_uses_mr_diff_refs_for_valid_diff_line(self, gitlab_provider):
        gitlab_provider.mr = MagicMock()
        gitlab_provider.mr.diff_refs = {
            "base_sha": "base-ref",
            "start_sha": "start-ref",
            "head_sha": "head-ref",
        }
        gitlab_provider.diff_files = [
            FilePatchInfo(
                base_file="line old",
                head_file="line new",
                patch="@@ -1,1 +1,1 @@\n-line old\n+line new",
                filename="src/app.py",
            )
        ]
        suggestion = {
            "body": "**Suggestion:** Use better text\n```suggestion\nline better\n```",
            "relevant_file": "src/app.py",
            "relevant_lines_start": 1,
            "relevant_lines_end": 1,
            "original_suggestion": {
                "relevant_lines_start": 1,
                "relevant_lines_end": 1,
                "existing_code": "line new",
                "improved_code": "line better",
                "suggestion_content": "Use better text",
                "label": "possible issue",
                "score": 7,
            },
        }

        with patch.object(gitlab_provider, "get_diff_files", return_value=gitlab_provider.diff_files), \
             patch("pr_agent.git_providers.gitlab_provider.get_logger"):
            gitlab_provider.publish_code_suggestions([suggestion])

        gitlab_provider.mr.discussions.create.assert_called_once()
        position = gitlab_provider.mr.discussions.create.call_args.args[0]["position"]

        assert position["base_sha"] == "base-ref"
        assert position["start_sha"] == "start-ref"
        assert position["head_sha"] == "head-ref"
        assert position["new_line"] == 1
        gitlab_provider.mr.notes.create.assert_not_called()
