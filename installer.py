import os
import shutil
import signal
import subprocess
import sys
from argparse import ArgumentParser, HelpFormatter

# Assuming facefusion, metadata, wording are in the same project structure
# If not, you might need to adjust imports or ensure they are in PYTHONPATH
try:
    from facefusion import metadata, wording
    from facefusion.common_helper import is_linux, is_windows
except ImportError:
    # Fallback for direct script execution if not installed as a package
    # This might require 'facefusion' directory to be in sys.path or current dir
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        from facefusion import metadata, wording
        from facefusion.common_helper import is_linux, is_windows
    except ImportError:
        # Mocking for standalone execution if imports are problematic
        # In a real scenario, ensure your project structure allows these imports
        print("Warning: facefusion imports failed. Using mock objects for metadata and wording.", file=sys.stderr)
        class MockWording:
            def get(self, key):
                return f"{{{key}}}"
        class MockMetadata:
            def get(self, key):
                if key == 'name': return "facefusion_mock"
                if key == 'version': return "0.0.0"
                return "mock_value"
        wording = MockWording()
        metadata = MockMetadata()
        # Mock platform checks if common_helper is also missing
        is_windows = lambda: os.name == 'nt'
        is_linux = lambda: sys.platform.startswith('linux')


ONNXRUNTIME_SET =\
{
	'default': ('onnxruntime', '1.21.1')
}
if is_windows() or is_linux():
	ONNXRUNTIME_SET['cuda'] = ('onnxruntime-gpu', '1.21.1')
	ONNXRUNTIME_SET['openvino'] = ('onnxruntime-openvino', '1.21.0')
if is_windows():
	ONNXRUNTIME_SET['directml'] = ('onnxruntime-directml', '1.17.3')
if is_linux():
	ONNXRUNTIME_SET['rocm'] = ('onnxruntime-rocm', '1.21.0')


def check_uv_installed() -> str:
    uv_executable = shutil.which('uv')
    if uv_executable is None:
        sys.stderr.write(f"Error: 'uv' command not found. Please install uv first.{os.linesep}")
        sys.stderr.write(f"Visit https://github.com/astral-sh/uv for installation instructions.{os.linesep}")
        sys.exit(1)
    return uv_executable


def cli() -> None:
	signal.signal(signal.SIGINT, lambda signal_number, frame: sys.exit(0))
	program = ArgumentParser(formatter_class = lambda prog: HelpFormatter(prog, max_help_position = 50))
	program.add_argument('--onnxruntime', help = wording.get('help.install_dependency').format(dependency = 'onnxruntime'), choices = ONNXRUNTIME_SET.keys(), required = True)
	program.add_argument('--skip-conda', help = wording.get('help.skip_conda'), action = 'store_true')
	program.add_argument('-v', '--version', version = metadata.get('name') + ' ' + metadata.get('version'), action = 'version')
	run(program)


def run(program : ArgumentParser) -> None:
	args = program.parse_args()
	uv_executable = check_uv_installed()

	has_conda = 'CONDA_PREFIX' in os.environ
	onnxruntime_name, onnxruntime_version = ONNXRUNTIME_SET.get(args.onnxruntime)

	if not args.skip_conda and not has_conda:
		sys.stdout.write(wording.get('conda_not_activated') + os.linesep)
		sys.exit(1)

	print(f"Installing dependencies from requirements.txt using '{uv_executable} pip install'...")
	try:
		with open('requirements.txt') as file:
			for line in file:
				__line__ = line.strip()
				if __line__ and not __line__.startswith('#') and not __line__.startswith('onnxruntime'):
					# Pass the whole line to uv, it can handle comments and other directives like pip
					# Splitting by space can break hashes or other complex directives
					print(f"Installing: {__line__}")
					# Using subprocess.run for better control and error handling
					subprocess.run([uv_executable, 'pip', 'install', __line__, '--reinstall'], check=True)
	except FileNotFoundError:
		sys.stderr.write(f"Error: requirements.txt not found in the current directory.{os.linesep}")
		sys.exit(1)
	except subprocess.CalledProcessError as e:
		sys.stderr.write(f"Error installing dependency from requirements.txt: {e}{os.linesep}")
		sys.exit(1)


	print(f"Installing onnxruntime variant: {args.onnxruntime}")
	try:
		if args.onnxruntime == 'rocm':
			python_id = 'cp' + str(sys.version_info.major) + str(sys.version_info.minor)
			# Note: Original script had rocm_rel_6.4. The version in ONNXRUNTIME_SET is 1.21.0.
			# For rocm 1.21.0, the repo might be different e.g. rocm-rel-6.1.2 or 6.0.
			# This URL might need adjustment based on the actual onnxruntime-rocm version.
			# Example: https://repo.radeon.com/rocm/manylinux/rocm-rel-6.0/
			# The original script used onnxruntime_version (1.21.0) with rocm-rel-6.4, which might be a mismatch.
			# Let's assume the version from ONNXRUNTIME_SET (1.21.0) is the driver.
			# Finding the exact ROCm release for onnxruntime-rocm 1.21.0:
			# Typically, ORT ROCm releases are tied to specific ROCm toolkit versions.
			# For 1.21.0, it might be an older ROCm release. Checking ORT docs is best.
			# For now, let's keep a placeholder or a known working one if available.
			# The original script had onnxruntime_version = '1.21.0' for rocm and used 'rocm-rel-6.4'.
			# This could be correct if 1.21.0 was built against 6.4.
			# Let's stick to the original logic for the URL structure.
			rocm_release_path = 'rocm-rel-6.4' # As per original script context
			# If onnxruntime_version for rocm changes in ONNXRUNTIME_SET, rocm_release_path might also need to change.

			if python_id in [ 'cp310', 'cp311', 'cp312' ]: # Adjusted for typical supported versions
				wheel_name = f'onnxruntime_rocm-{onnxruntime_version}-{python_id}-{python_id}-linux_x86_64.whl'
				wheel_url = f'https://repo.radeon.com/rocm/manylinux/{rocm_release_path}/{wheel_name}'
				print(f"Installing ROCm ONNXRuntime from: {wheel_url}")
				subprocess.run([uv_executable, 'pip', 'install', wheel_url, '--reinstall'], check=True)
			else:
				sys.stderr.write(f"Warning: Python version {python_id} may not have a pre-built ROCm ONNXRuntime wheel at the default URL.{os.linesep}")
				sys.stderr.write(f"Attempting standard install for {onnxruntime_name}=={onnxruntime_version}, which might fail or not use ROCm.{os.linesep}")
				subprocess.run([uv_executable, 'pip', 'install', f'{onnxruntime_name}=={onnxruntime_version}', '--reinstall'], check=True)

		else:
			print(f"Installing: {onnxruntime_name}=={onnxruntime_version}")
			subprocess.run([uv_executable, 'pip', 'install', f'{onnxruntime_name}=={onnxruntime_version}', '--reinstall'], check=True)
	except subprocess.CalledProcessError as e:
		sys.stderr.write(f"Error installing onnxruntime: {e}{os.linesep}")
		sys.exit(1)


	if args.onnxruntime == 'cuda' and has_conda and not args.skip_conda:
		conda_executable = shutil.which('conda')
		if not conda_executable:
			sys.stdout.write(f"Warning: 'conda' executable not found. Skipping Conda environment variable setup for CUDA.{os.linesep}")
		else:
			print("Configuring Conda environment variables for CUDA...")
			library_paths = []
			env_var_name = ''

			if is_linux():
				env_var_name = 'LD_LIBRARY_PATH'
				if os.getenv(env_var_name):
					library_paths = os.getenv(env_var_name).split(os.pathsep)
				python_id = 'python' + str(sys.version_info.major) + '.' + str(sys.version_info.minor)
				conda_prefix = os.getenv('CONDA_PREFIX')
				potential_paths = [
					os.path.join(conda_prefix, 'lib'),
					os.path.join(conda_prefix, 'lib', python_id, 'site-packages', 'tensorrt_libs')
				]
				library_paths.extend(potential_paths)

			if is_windows():
				env_var_name = 'PATH'
				if os.getenv(env_var_name):
					library_paths = os.getenv(env_var_name).split(os.pathsep)
				conda_prefix = os.getenv('CONDA_PREFIX')
				potential_paths = [
					os.path.join(conda_prefix, 'Lib', 'site-packages', 'nvidia', 'cudnn', 'bin'), # More specific path for cudnn
					os.path.join(conda_prefix, 'Lib', 'site-packages', 'tensorrt_libs'),
					os.path.join(conda_prefix, 'Library', 'bin') # General conda bin for DLLs
				]
				library_paths.extend(potential_paths)

			# Deduplicate and filter existing paths
			valid_library_paths = []
			seen_paths = set()
			for path in library_paths:
				if path and os.path.exists(path) and path not in seen_paths:
					valid_library_paths.append(path)
					seen_paths.add(path)

			if valid_library_paths and env_var_name:
				try:
					print(f"Setting {env_var_name} for Conda environment: {os.pathsep.join(valid_library_paths)}")
					subprocess.run([conda_executable, 'env', 'config', 'vars', 'set', f'{env_var_name}={os.pathsep.join(valid_library_paths)}'], check=True)
					sys.stdout.write(f"{os.linesep}Conda environment variables set. Please reactivate your Conda environment for changes to take effect.{os.linesep}")
					sys.stdout.write(f"Example: conda activate {os.getenv('CONDA_DEFAULT_ENV')}{os.linesep}")
				except subprocess.CalledProcessError as e:
					sys.stderr.write(f"Error setting Conda environment variables: {e}{os.linesep}")
			else:
				sys.stdout.write(f"No valid library paths found or environment variable name not set for CUDA Conda setup.{os.linesep}")


	if args.onnxruntime == 'directml':
		print("Installing specific numpy version for DirectML ONNXRuntime...")
		try:
			subprocess.run([uv_executable, 'pip', 'install', 'numpy==1.26.4', '--reinstall'], check=True)
		except subprocess.CalledProcessError as e:
			sys.stderr.write(f"Error installing numpy for DirectML: {e}{os.linesep}")
			sys.exit(1)

	sys.stdout.write(f"{os.linesep}Installation process completed.{os.linesep}")


if __name__ == '__main__':
	# This check ensures that the script can be run directly
	# and the imports for facefusion.metadata etc. might need adjustment
	# depending on your project structure.
	# If 'facefusion' is a package you install, this script should ideally
	# be an entry point defined in setup.py/pyproject.toml.
	cli()
