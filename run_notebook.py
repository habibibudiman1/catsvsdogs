from pathlib import Path
from datetime import datetime, timezone
import json
import os
import re
import sys
import time

project_dir = Path(__file__).resolve().parent
runtime_dir = project_dir / '.tools' / 'jupyter'
runtime_dir.mkdir(parents=True, exist_ok=True)
os.environ['JUPYTER_PATH'] = str(runtime_dir)
os.environ['JUPYTER_RUNTIME_DIR'] = str(runtime_dir / 'runtime')
os.environ['IPYTHONDIR'] = str(project_dir / '.tools' / 'ipython')
os.environ['MPLCONFIGDIR'] = str(project_dir / '.tools' / 'mplconfig')
os.environ['KERAS_HOME'] = str(project_dir / '.tools' / 'keras_home')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = str(min(6, os.cpu_count() or 2))
os.environ['TF_NUM_INTRAOP_THREADS'] = os.environ['OMP_NUM_THREADS']
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
os.environ['PYTHONUNBUFFERED'] = '1'

import nbformat
from nbclient import NotebookClient

kernel_name = 'catvsdog-project'
kernel_dir = runtime_dir / 'kernels' / kernel_name
kernel_dir.mkdir(parents=True, exist_ok=True)
kernel_spec = {
    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'Cats vs Dogs (project)',
    'language': 'python',
}
(kernel_dir / 'kernel.json').write_text(json.dumps(kernel_spec), encoding='utf-8')

notebook_name = 'cats_vs_dogs_indonesia.ipynb'
if len(sys.argv) > 1:
    notebook_name = sys.argv[1]
source_path = (project_dir / notebook_name).resolve()
if not source_path.is_relative_to(project_dir) or source_path.suffix != '.ipynb':
    raise ValueError('Pilih file notebook di dalam folder proyek.')
output_path = source_path.with_name(source_path.stem + '.executed.ipynb')
status_path = runtime_dir / 'execution_status.json'
log_path = runtime_dir / 'execution.log'
notebook = nbformat.read(source_path, as_version=4)
for cell in notebook.cells:
    if cell.cell_type == 'code':
        cell.outputs = []
        cell.execution_count = None
nbformat.validate(notebook)
ansi_pattern = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
status = {
    'state': 'starting', 'source': str(source_path), 'output': str(output_path),
    'runner_pid': os.getpid(),
}
last_saved_at = 0.0


def save_progress():
    global last_saved_at
    status['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
    status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')
    temporary_path = output_path.with_suffix('.tmp.ipynb')
    nbformat.write(notebook, temporary_path)
    temporary_path.replace(output_path)
    last_saved_at = time.monotonic()


def start_cell(cell, cell_index):
    if cell.cell_type == 'code':
        status['state'] = 'running'
        status['cell_id'] = cell.get('id', str(cell_index))
        status['cell_index'] = cell_index
        print('EXECUTING', status['cell_id'], flush=True)
        save_progress()


def finish_cell(cell, cell_index, execute_reply):
    save_progress()


class ProgressClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        output = super().process_message(msg, cell, cell_index)
        if msg['msg_type'] == 'stream':
            stream_text = ansi_pattern.sub('', msg['content']['text']).replace('\b', '')
            with log_path.open('a', encoding='utf-8') as log_file:
                log_file.write(stream_text.replace('\r', '\n'))
            status['latest_output'] = stream_text[-1200:]
            for line in stream_text.splitlines():
                if line.startswith('Hasil: '):
                    status['run_dir'] = line[len('Hasil: '):].strip()
        if time.monotonic() - last_saved_at >= 30:
            save_progress()
        return output


client = ProgressClient(
    notebook,
    timeout=None,
    startup_timeout=180,
    iopub_timeout=120,
    kernel_name=kernel_name,
    ipython_hist_file=':memory:',
    allow_errors=False,
    on_cell_start=start_cell,
    on_cell_executed=finish_cell,
)
save_progress()
try:
    client.execute(cwd=str(project_dir))
except BaseException as error:
    status['state'] = 'failed'
    status['error'] = str(error)[-4000:]
    save_progress()
    raise
else:
    status['state'] = 'completed'
    save_progress()
    print('COMPLETED', output_path, flush=True)
