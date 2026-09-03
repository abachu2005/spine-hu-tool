"""Unit tests for the scout body-habitus measurement.

Everything here runs on synthetic projections with a known right answer, so the
geometry, the couch rejection, and the magnification correction are all checked
without needing DICOM or a segmentation model.
"""
import numpy as np
import pytest

from spine_hu_tool.config import ScoutParams
from spine_hu_tool.scout.loader import ScoutImage, load_scout
from spine_hu_tool.scout.thickness import (body_profile, measure_habitus,
                                           level_z_ranges, scout_json)
from spine_hu_tool.tests.synthetic import make_scout_pair

LEVELS = {"L1": (-20.0, 20.0)}


def _measure(**kw):
    ap, lat = make_scout_pair(**kw)
    return measure_habitus([ap, lat], LEVELS)["levels"]["L1"]


# --- geometry and outline ------------------------------------------------
def test_recovers_known_torso_dimensions():
    # Centered patient, no divergence: the measurement must return the phantom's
    # own dimensions. A threshold on a smooth chord profile clips the very tip of
    # the ellipse, so a fraction of a percent of under-read is expected.
    v = _measure(width_lr_mm=360.0, depth_ap_mm=280.0, center=(0.0, 0.0),
                 magnify=False, table=False)
    assert v["body_width_lr_mm"] == pytest.approx(360.0, rel=0.02)
    assert v["body_depth_ap_mm"] == pytest.approx(280.0, rel=0.02)
    assert v["body_effective_diameter_mm"] == pytest.approx(
        np.sqrt(360.0 * 280.0), rel=0.02)
    assert v["scout_warnings"] == []


def test_couch_does_not_inflate_the_lateral_depth():
    common = dict(width_lr_mm=360.0, depth_ap_mm=280.0, center=(0.0, 0.0),
                  magnify=False)
    with_table = _measure(table=True, **common)
    without = _measure(table=False, **common)
    # The couch adds ~47 mm of material posterior to the patient; if it leaked
    # into the outline the depth would jump by that much.
    assert with_table["body_depth_ap_mm"] == pytest.approx(
        without["body_depth_ap_mm"], abs=1.0)
    assert with_table["body_width_lr_mm"] == pytest.approx(
        without["body_width_lr_mm"], abs=1.0)


def test_view_classification_and_axes():
    ap, lat = make_scout_pair()
    assert (ap.view, ap.axis_name, ap.beam_axis) == ("AP", "x", 1)
    assert (lat.view, lat.axis_name, lat.beam_axis) == ("LAT", "y", 0)


def test_row_and_z_map_round_trip():
    ap, _ = make_scout_pair(z_range=(-200.0, 200.0))
    rows = np.array([0, 37, 250])
    assert np.allclose(ap.z_to_row(ap.row_to_z(rows)), rows)
    lo, hi = ap.z_range
    assert (lo, hi) == pytest.approx((-200.0, 200.0), abs=1.0)


def test_profile_boundaries_are_symmetric_about_the_body_center():
    ap, _ = make_scout_pair(width_lr_mm=340.0, center=(25.0, 0.0),
                            magnify=False, table=False)
    prof = body_profile(ap)
    mid = np.nanmedian(prof["center_mm"])
    assert mid == pytest.approx(25.0, abs=2.0)
    assert np.nanmedian(prof["width_mm"]) == pytest.approx(340.0, rel=0.02)


# --- magnification --------------------------------------------------------
def test_magnification_correction_recovers_truth():
    # An off-isocenter patient projects at the wrong scale; each view supplies
    # the offset the other one needs, so the pair must recover both dimensions.
    kw = dict(width_lr_mm=360.0, depth_ap_mm=280.0, center=(15.0, 40.0),
              table=False)
    v = _measure(magnify=True, **kw)
    assert v["body_width_lr_mm"] == pytest.approx(360.0, rel=0.02)
    assert v["body_depth_ap_mm"] == pytest.approx(280.0, rel=0.02)
    # The uncorrected numbers are the ones that are wrong, so the correction is
    # demonstrably doing work rather than being a no-op.
    assert v["body_width_lr_uncorrected_mm"] < 350.0
    assert v["scout_ap_magnification"] == pytest.approx(580.0 / 540.0, rel=0.01)
    assert v["scout_lat_magnification"] == pytest.approx(555.0 / 540.0, rel=0.01)


def test_magnification_is_identity_for_a_centered_patient():
    v = _measure(center=(0.0, 0.0), magnify=True, table=False)
    assert v["scout_ap_magnification"] == pytest.approx(1.0, abs=0.01)
    assert v["scout_lat_magnification"] == pytest.approx(1.0, abs=0.01)


def test_correction_is_skipped_without_source_geometry():
    ap, lat = make_scout_pair(center=(0.0, 40.0), magnify=False, table=False)
    ap.sod_mm = lat.sod_mm = None
    v = measure_habitus([ap, lat], LEVELS)["levels"]["L1"]
    assert v["scout_ap_magnification"] == 1.0
    assert v["body_width_lr_mm"] == v["body_width_lr_uncorrected_mm"]


# --- QC and degradation ---------------------------------------------------
def test_missing_scouts_degrade_gracefully():
    out = measure_habitus([], LEVELS)
    assert out["available"] is False
    assert out["levels"] == {}
    assert out["warnings"] and "no scout" in out["warnings"][0]


def test_single_view_reports_only_what_it_measured():
    ap, _ = make_scout_pair(magnify=False, table=False)
    out = measure_habitus([ap], LEVELS)
    v = out["levels"]["L1"]
    assert "body_width_lr_mm" in v
    assert "body_depth_ap_mm" not in v
    assert "body_effective_diameter_mm" not in v
    assert any("lateral" in w for w in out["warnings"])


def test_implausible_extent_is_flagged_not_silently_reported():
    # A torso wider than any patient (arms/shoulders projected into the field).
    v = _measure(width_lr_mm=520.0, depth_ap_mm=280.0, center=(0.0, 0.0),
                 magnify=False, table=False)
    assert any("plausible" in w or "edge" in w for w in v["scout_warnings"])


def test_unstable_outline_across_a_level_is_flagged():
    # Rows within one vertebra are millimeters apart, so a large row-to-row jump
    # means the outline left the skin (typically onto an arm), not that the
    # patient changed shape. The reported value stays the band median.
    ap, lat = make_scout_pair(width_lr_mm=360.0, center=(0.0, 0.0),
                              magnify=False, table=False)
    peak = ap.pixels.max()
    rows = ap.z_to_row(np.array([4.0, 20.0])).astype(int)
    edge = int(np.argmax(ap.pixels[0, :] > ap.pixels.min() + 10))
    # An arm resting against the flank: contiguous with the torso, so it joins
    # the detected run instead of being rejected as a separate object.
    ap.pixels[min(rows):max(rows), edge - 90:edge + 20] = peak
    v = measure_habitus([ap, lat], {"L1": (-20.0, 20.0)})["levels"]["L1"]
    assert any("varies by" in w for w in v["scout_warnings"])
    assert v["body_width_lr_mm"] == pytest.approx(360.0, rel=0.03)


def test_shoulder_girdle_levels_say_so():
    ap, lat = make_scout_pair(magnify=False, table=False)
    out = measure_habitus([ap, lat], {"T1": (-20.0, 20.0), "T8": (-20.0, 20.0)})
    assert any("shoulder" in w for w in out["levels"]["T1"]["scout_warnings"])
    assert not any("shoulder" in w for w in out["levels"]["T8"]["scout_warnings"])


def test_off_isocenter_patient_is_flagged():
    v = _measure(center=(0.0, 90.0), magnify=False, table=False)
    assert any("isocenter" in w for w in v["scout_warnings"])


def test_scout_json_drops_cached_arrays():
    ap, lat = make_scout_pair()
    block = measure_habitus([ap, lat], LEVELS)
    assert "_scouts" in block and "_profiles" in block
    safe = scout_json(block)
    assert not any(k.startswith("_") for k in safe)
    import json
    json.dumps(safe)          # must be serializable as-is


# --- level mapping --------------------------------------------------------
def test_level_z_ranges_uses_the_volume_origin():
    from spine_hu_tool.core import Volume
    from spine_hu_tool.segmentation.totalseg_runner import label_id_for
    seg = np.zeros((8, 8, 40), dtype=np.int16)
    seg[3:5, 3:5, 10:20] = label_id_for("L1")
    vol = Volume(hu=np.zeros_like(seg, dtype=np.float32),
                 spacing=(1.0, 1.0, 2.0), origin=(0.0, 0.0, -100.0))
    ranges = level_z_ranges(seg, vol)
    assert ranges["L1"] == pytest.approx((-80.0, -62.0))


def test_level_band_averages_the_center_of_the_level():
    ap, lat = make_scout_pair(magnify=False, table=False)
    out = measure_habitus([ap, lat], {"L1": (-30.0, 30.0)},
                          ScoutParams(level_band_frac=0.5))
    band = out["levels"]["L1"]["scout_band_z_mm"]
    assert band == pytest.approx([-15.0, 15.0], abs=0.1)


# --- DICOM loading --------------------------------------------------------
def _write_localizer(path, iop, ipp, rows=64, cols=48):
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    ds.SOPClassUID = pydicom.uid.CTImageStorage
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.ImageType = ["ORIGINAL", "PRIMARY", "LOCALIZER"]
    ds.ImageOrientationPatient = list(iop)
    ds.ImagePositionPatient = list(ipp)
    ds.PixelSpacing = [0.5, 0.6]
    ds.Rows, ds.Columns = rows, cols
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
    ds.PixelRepresentation = 0
    ds.RescaleSlope, ds.RescaleIntercept = 1.0, -1024.0
    ds.DistanceSourceToDetector, ds.DistanceSourceToPatient = 950.0, 540.0
    ds.TableHeight = 150.0
    ds.PixelData = np.full((rows, cols), 600, dtype=np.uint16).tobytes()
    ds.save_as(path, enforce_file_format=True)
    return path


def test_load_scout_classifies_ap_and_lateral(tmp_path):
    ap = load_scout(_write_localizer(str(tmp_path / "ap.dcm"),
                                     [1, 0, 0, 0, 0, -1], [-100, 0, 200]))
    lat = load_scout(_write_localizer(str(tmp_path / "lat.dcm"),
                                      [0, -1, 0, 0, 0, -1], [0, 100, 200]))
    assert ap.view == "AP" and ap.axis == 0 and ap.beam_axis == 1
    assert lat.view == "LAT" and lat.axis == 1 and lat.beam_axis == 0
    assert ap.sod_mm == 540.0 and ap.table_height_mm == 150.0
    # rescale must be applied, otherwise every threshold would be off by 1024
    assert ap.pixels.min() == pytest.approx(600 - 1024)
    # rows run inferiorly (col_dir = -z), so row 0 is the most superior
    assert ap.row_to_z(0) == pytest.approx(200.0)
    assert ap.row_to_z(1) < ap.row_to_z(0)


def test_load_scout_rejects_a_non_localizer_geometry(tmp_path):
    # columns running along z is not a projection this module can interpret
    assert load_scout(_write_localizer(str(tmp_path / "odd.dcm"),
                                       [0, 0, 1, 0, 1, 0], [0, 0, 0])) is None
