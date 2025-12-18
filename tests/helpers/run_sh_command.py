
import pytest

from tests.helpers.package_available import _SH_AVAILABLE

if _SH_AVAILABLE:
    import sh


def run_sh_command(command: list[str]):
    """Default method for executing shell commands with pytest and sh
    package."""
    msg = None
    try:
        sh.python(command, _err_to_out=True)
    except sh.ErrorReturnCode as e:
        stderr = e.stderr.decode() if e.stderr else ''
        # Filter out known warnings that don't indicate test failure
        filtered_lines = []
        for line in stderr.split('\n'):
            # Skip pynvml deprecation warnings and pkg_resources warnings
            if 'pynvml package is deprecated' in line:
                continue
            if 'pkg_resources is deprecated' in line:
                continue
            if 'FutureWarning' in line and 'pynvml' in line:
                continue
            if 'UserWarning' in line and 'pkg_resources' in line:
                continue
            filtered_lines.append(line)
        msg = '\n'.join(filtered_lines).strip()
    if msg:
        pytest.fail(msg)
