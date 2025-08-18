import os

# Reproducibility
RANDOM_SEED: int = 42
EPS: float = 1e-6

# Relative paths (competition rule)
DATA_DIR: str = os.path.join("data")
OUTPUT_DIR: str = os.path.join("outputs")
MODELS_DIR: str = os.path.join(OUTPUT_DIR, "models")
LOGS_DIR: str = os.path.join(OUTPUT_DIR, "logs")
SUBMISSIONS_DIR: str = os.path.join(OUTPUT_DIR, "submissions")

# Column name mapping (Korean headers -> internal standard)
COLUMN_MAP = {
	"건물번호": "building_id",
	"일시": "timestamp",
	"전력소비량(kWh)": "load",
	"기온(°C)": "temp",
	"강수량(mm)": "rain",
	"풍속(m/s)": "wind",
	"습도(%)": "humid",
	"일조(hr)": "sunshine",
	"일사(MJ/m2)": "irradiance",
}

# Building info mapping
BUILDING_INFO_COLUMN_MAP = {
	"건물번호": "building_id",
	"건물유형": "type",
	"연면적(m2)": "total_area",
	"냉방면적(m2)": "cooling_area",
	"태양광용량(kW)": "pv_capacity",
	"ESS저장용량(kWh)": "ess_capacity",
	"PCS용량(kW)": "pcs_capacity",
}
