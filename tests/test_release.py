import hashlib
import json
import zipfile

import pytest

from prepare_release import prepare


def test_neural_archive_is_complete_hashed_deterministic_and_non_overwriting(tmp_path):
    first, second = tmp_path/'first.zip', tmp_path/'second.zip'
    record = prepare(first)
    prepare(second)
    assert first.read_bytes() == second.read_bytes()
    assert record['version'] == '0.4.0'
    assert record['archive_sha256'] == hashlib.sha256(first.read_bytes()).hexdigest()
    with zipfile.ZipFile(first) as archive:
        prefix = 'advg-reference/'
        manifest = json.loads(archive.read(prefix+'SOURCE_MANIFEST.json'))
        assert len(archive.namelist()) == len(manifest['sha256']) + 1
        for name, digest in manifest['sha256'].items():
            assert hashlib.sha256(archive.read(prefix+name)).hexdigest() == digest
        assert prefix+'verification/single_value_update_fixtures.npz' in archive.namelist()
        for recipe in ('algorithm', 'lq', 'pendulum'):
            assert prefix+'configs/'+recipe+'.json' in archive.namelist()
        assert not any('/runs/' in name or '__pycache__' in name or name.endswith('.pt')
                       for name in archive.namelist())
    with pytest.raises(FileExistsError):
        prepare(first)
