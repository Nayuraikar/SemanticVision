import json
from core.perception.export_ts import FIXTURE, TS_OUT, fixture, ts_source


def test_cvd_exports_in_sync():
    assert TS_OUT.read_text() == ts_source(), "run `python -m core.perception.export_ts`"
    assert json.loads(FIXTURE.read_text()) == json.loads(json.dumps(fixture()))
