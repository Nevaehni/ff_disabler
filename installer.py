import os
import shutil
import signal
import subprocess
import sys
from argparse import ArgumentParser, HelpFormatter
from functools import partial
from types import FrameType
from facefusion import metadata, wording
from facefusion.common_helper import is_linux, is_windows

ONNXRUNTIME_SET =\
{
	'default': ('onnxruntime', '1.22.0')
}
if is_windows() or is_linux():
	ONNXRUNTIME_SET['cuda'] = ('onnxruntime-gpu', '1.22.0')
	ONNXRUNTIME_SET['openvino'] = ('onnxruntime-openvino', '1.22.0')
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
	signal.signal(signal.SIGINT, signal_exit)
	program = ArgumentParser(formatter_class = partial(HelpFormatter, max_help_position = 50))
	program.add_argument('--onnxruntime', help = wording.get('help.install_dependency').format(dependency = 'onnxruntime'), choices = ONNXRUNTIME_SET.keys(), required = True)
	program.add_argument('--skip-conda', help = wording.get('help.skip_conda'), action = 'store_true')
	program.add_argument('-v', '--version', version = metadata.get('name') + ' ' + metadata.get('version'), action = 'version')
	run(program)


def signal_exit(signum : int, frame : FrameType) -> None:
	sys.exit(0)


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
			for line in file.readlines():
				**line** = line.strip()
				if **line** and not **line**.startswith('#') and not **line**.startswith('onnxruntime'):
					print(f"Installing: {**line**}")
					subprocess.call([uv_executable, 'pip', 'install', **line**, '--force-reinstall'])
	except FileNotFoundError:
		sys.stderr.write(f"Error: requirements.txt not found in the current directory.{os.linesep}")
		sys.exit(1)
	
	print(f"Installing onnxruntime variant: {args.onnxruntime}")
	if args.onnxruntime == 'rocm':
		python_id = 'cp' + str(sys.version_info.major) + str(sys.version_info.minor)
		if python_id in [ 'cp310', 'cp312' ]:
			wheel_name = 'onnxruntime_rocm-' + onnxruntime_version + '-' + python_id + '-' + python_id + '-linux_x86_64.whl'
			wheel_url = 'https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4/' + wheel_name
			print(f"Installing ROCm ONNXRuntime from: {wheel_url}")
			subprocess.call([uv_executable, 'pip', 'install', wheel_url, '--force-reinstall'])
		else:
			sys.stderr.write(f"Warning: Python version {python_id} may not have a pre-built ROCm ONNXRuntime wheel.{os.linesep}")
			sys.stderr.write(f"Attempting standard install for {onnxruntime_name}=={onnxruntime_version}.{os.linesep}")
			subprocess.call([uv_executable, 'pip', 'install', f'{onnxruntime_name}=={onnxruntime_version}', '--force-reinstall'])
	else:
		print(f"Installing: {onnxruntime_name}=={onnxruntime_version}")
		subprocess.call([uv_executable, 'pip', 'install', onnxruntime_name + '==' + onnxruntime_version, '--force-reinstall'])
	
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
				library_paths.extend([
					os.path.join(os.getenv('CONDA_PREFIX'), 'lib'),
					os.path.join(os.getenv('CONDA_PREFIX'), 'lib', python_id, 'site-packages', 'tensorrt_libs')
				])
			
			if is_windows():
				env_var_name = 'PATH'
				if os.getenv(env_var_name):
					library_paths = os.getenv(env_var_name).split(os.pathsep)
				library_paths.extend([
					os.path.join(os.getenv('CONDA_PREFIX'), 'Lib', 'site-packages', 'nvidia', 'cudnn', 'bin'),
					os.path.join(os.getenv('CONDA_PREFIX'), 'Lib', 'site-packages', 'tensorrt_libs'),
					os.path.join(os.getenv('CONDA_PREFIX'), 'Library', 'bin')
				])
			
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
					subprocess.call([conda_executable, 'env', 'config', 'vars', 'set', f'{env_var_name}={os.pathsep.join(valid_library_paths)}'])
					sys.stdout.write(f"{os.linesep}Conda environment variables set. Please reactivate your Conda environment for changes to take effect.{os.linesep}")
					sys.stdout.write(f"Example: conda activate {os.getenv('CONDA_DEFAULT_ENV')}{os.linesep}")
				except Exception as e:
					sys.stderr.write(f"Error setting Conda environment variables: {e}{os.linesep}")
			else:
				sys.stdout.write(f"No valid library paths found or environment variable name not set for CUDA Conda setup.{os.linesep}")
	
	if args.onnxruntime == 'directml':
		print("Installing specific numpy version for DirectML ONNXRuntime...")
		subprocess.call([uv_executable, 'pip', 'install', 'numpy==1.26.4', '--force-reinstall'])
	
	sys.stdout.write(f"{os.linesep}Installation process completed.{os.linesep}")


if __name__ == '__main__':
	cli()
