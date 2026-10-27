from __future__ import annotations

import importlib.util
import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


COLAB = Path(__file__).parents[1] / "colab"
sys.path.insert(0, str(COLAB))  # run_host_native imports run_dance_swap the way it does on Colab
SPEC = importlib.util.spec_from_file_location("run_host_native", COLAB / "run_host_native.py")
assert SPEC and SPEC.loader
run_host_native = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_host_native
SPEC.loader.exec_module(run_host_native)


def archive(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as bundle:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))
    return path


class UnpackTests(unittest.TestCase):
    def test_pipeline_is_unpacked_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dest = tmp / "h3-dub"
            (dest / "src").mkdir(parents=True)
            (dest / "src" / "stale.py").write_text("old", encoding="utf-8")
            bundle = archive(tmp / "p.tar.gz", {"src/workflow.py": b"x = 1\n", "config/a.yaml": b"a: 1\n"})
            run_host_native.unpack(bundle, dest)
            self.assertEqual((dest / "src" / "workflow.py").read_text(encoding="utf-8"), "x = 1\n")
            self.assertTrue((dest / "config" / "a.yaml").is_file())
            self.assertFalse((dest / "src" / "stale.py").exists())

    def test_member_outside_the_pipeline_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            bundle = archive(tmp / "p.tar.gz", {"src/ok.py": b"", "../escape.py": b"boom"})
            with self.assertRaises(SystemExit):
                run_host_native.unpack(bundle, tmp / "h3-dub")
            self.assertFalse((tmp / "escape.py").exists())


if __name__ == "__main__":
    unittest.main()
