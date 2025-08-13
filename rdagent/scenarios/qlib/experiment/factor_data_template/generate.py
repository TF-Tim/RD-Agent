import qlib
from pathlib import Path
from qlib.data import D


BUNDLE_DIR = Path("/root/.qlib/qlib_data/us_data")
BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

print("Initializing Qlib from standard container path…")
qlib.init(provider_uri=str(BUNDLE_DIR))

debug_start_date = "2024-01-01"
debug_end_date   = "2024-03-31"
fields = ["$open", "$close", "$high", "$low", "$volume"]

print("Loading all instruments from your data…")
instruments = D.instruments()

print("Loading full dataset to create daily_pv.h5 & daily_pv_all.h5…")
all_data = D.features(instruments, fields, freq="day").swaplevel().sort_index()

all_data.to_hdf(BUNDLE_DIR / "daily_pv.h5",     key="data") 
all_data.to_hdf(BUNDLE_DIR / "daily_pv_all.h5", key="data")
print("✅  daily_pv.h5 & daily_pv_all.h5 created successfully.")

print(f"Loading slice {debug_start_date} → {debug_end_date} for daily_pv_debug.h5…")
debug_data = (
    D.features(
        instruments, fields,
        start_time=debug_start_date, end_time=debug_end_date, freq="day"
    )
    .swaplevel()
    .sort_index()
)
debug_data.to_hdf(BUNDLE_DIR / "daily_pv_debug.h5", key="data")
print("✅  daily_pv_debug.h5 created successfully.")

template_dir = Path(__file__).parent / "factor_data_template"
template_dir.mkdir(parents=True, exist_ok=True)

all_data.to_hdf(template_dir / "daily_pv_all.h5",  key="data")
debug_data.to_hdf(template_dir / "daily_pv_debug.h5", key="data")
print("✅  daily_pv_all.h5 & daily_pv_debug.h5 copied to factor_data_template.")
