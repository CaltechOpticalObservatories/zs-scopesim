"""Expected-rate notebook helpers must leave science readout noise enabled."""
import ast
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
import pytest


@pytest.mark.parametrize('name', ['zshooter_performance', 'zshooter_spie2026'])
@pytest.mark.parametrize('interrupted', [False, True])
def test_temporary_noise_disabling_restores_original_flags(name, interrupted):
    notebook = json.loads((Path(__file__).parents[1]/'notebooks'/f'{name}.ipynb').read_text())
    cell = next(cell for cell in notebook['cells'] if cell['id'] == '69b881fd0e3a9023')
    tree = ast.parse(''.join(cell['source']))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'temporarily_disable_effects')
    namespace = {'contextmanager': contextmanager, 'get_effect': lambda train, name, active_only: train[name]}
    exec(compile(ast.Module(body=[function], type_ignores=[]), name, 'exec'), namespace)
    train = {'shot_noise': SimpleNamespace(include=True), 'readout_noise_selector': SimpleNamespace(include=True)}
    try:
        with namespace['temporarily_disable_effects'](train, train):
            assert all(not effect.include for effect in train.values())
            if interrupted:
                raise RuntimeError('interrupted readout')
    except RuntimeError:
        assert interrupted
    assert all(effect.include for effect in train.values())
