import glob
import pyxdf

xdf_files = glob.glob("data/raw/**/*.xdf", recursive=True)

for fpath in xdf_files:
    print(f"\n=== {fpath} ===")
    streams, _ = pyxdf.load_xdf(fpath)
    for stream in streams:
        name = stream["info"]["name"][0]
        ch_count = int(stream["info"]["channel_count"][0])
        print(f"  Stream: {name!r}  ({ch_count} channels)")
        # Channel labels are nested under desc > channels > channel > label
        try:
            channels = stream["info"]["desc"][0]["channels"][0]["channel"]
            labels = [ch["label"][0] for ch in channels]
            print(f"  Columns: {labels}")
        except (KeyError, TypeError, IndexError):
            print(f"  Columns: (no label metadata, {ch_count} unnamed channels)")
