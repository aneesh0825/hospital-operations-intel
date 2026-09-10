import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = (
    "data/processed/"
    "synthea_readmission_ml_dataset_v3.csv"
)

TARGET = "readmitted_30_days"

CANDIDATE_METRICS = [
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "respiratory_rate",
    "glucose",
]


# ============================================================
# LOAD DATA
# ============================================================

data = pd.read_csv(DATA_PATH)

print("=" * 72)
print("ADMITRA V4 CLINICAL METRIC CHECK")
print("=" * 72)

print()
print("Rows:")
print(f"{len(data):,}")

print()
print("Overall readmission rate:")
print(f"{data[TARGET].mean() * 100:.2f}%")


# ============================================================
# CHECK AVAILABLE METRICS
# ============================================================

print()
print("=" * 72)
print("CANDIDATE METRIC AVAILABILITY")
print("=" * 72)

available_metrics = []

for metric in CANDIDATE_METRICS:

    if metric not in data.columns:
        print()
        print(f"{metric}: NOT FOUND")
        continue

    available_metrics.append(metric)

    available = data[metric].notna().sum()
    missing = data[metric].isna().sum()

    print()
    print(metric.upper())
    print("-" * 72)

    print(
        f"Available: {available:,} "
        f"({available / len(data) * 100:.2f}%)"
    )

    print(
        f"Missing: {missing:,} "
        f"({missing / len(data) * 100:.2f}%)"
    )

    print(f"Minimum: {data[metric].min():.2f}")
    print(f"Median: {data[metric].median():.2f}")
    print(f"Mean: {data[metric].mean():.2f}")
    print(f"Maximum: {data[metric].max():.2f}")

    correlation = data[metric].corr(data[TARGET])

    print(
        "Correlation with readmission: "
        f"{correlation:.4f}"
    )


# ============================================================
# READMISSION RATE ACROSS METRIC RANGES
# ============================================================

print()
print("=" * 72)
print("READMISSION RATE ACROSS METRIC RANGES")
print("=" * 72)

for metric in available_metrics:

    subset = data[
        [metric, TARGET]
    ].dropna().copy()

    print()
    print(metric.upper())
    print("-" * 72)

    try:

        subset["group"] = pd.qcut(
            subset[metric],
            q=5,
            duplicates="drop",
        )

        summary = (
            subset
            .groupby(
                "group",
                observed=False,
            )[TARGET]
            .agg(
                count="count",
                readmission_rate="mean",
            )
        )

        summary[
            "readmission_rate_percent"
        ] = (
            summary["readmission_rate"] * 100
        ).round(2)

        print(
            summary[
                [
                    "count",
                    "readmission_rate_percent",
                ]
            ]
        )

    except ValueError as error:

        print(
            "Could not create groups:",
            error,
        )


# ============================================================
# DERIVED METRICS
# ============================================================

print()
print("=" * 72)
print("DERIVED METRIC CANDIDATES")
print("=" * 72)


# ------------------------------------------------------------
# Pulse pressure
# systolic BP - diastolic BP
# ------------------------------------------------------------

if (
    "systolic_bp" in data.columns
    and "diastolic_bp" in data.columns
):

    data["pulse_pressure"] = (
        data["systolic_bp"]
        - data["diastolic_bp"]
    )

    print()
    print("PULSE PRESSURE")
    print("-" * 72)

    print(
        "Available:",
        f"{data['pulse_pressure'].notna().sum():,}",
    )

    print(
        "Median:",
        round(
            data["pulse_pressure"].median(),
            2,
        ),
    )

    print(
        "Correlation with readmission:",
        round(
            data["pulse_pressure"].corr(
                data[TARGET]
            ),
            4,
        ),
    )


# ------------------------------------------------------------
# Mean arterial pressure
#
# MAP ≈ DBP + 1/3(SBP - DBP)
# ------------------------------------------------------------

if (
    "systolic_bp" in data.columns
    and "diastolic_bp" in data.columns
):

    data["mean_arterial_pressure"] = (
        data["diastolic_bp"]
        + (
            data["systolic_bp"]
            - data["diastolic_bp"]
        ) / 3
    )

    print()
    print("MEAN ARTERIAL PRESSURE")
    print("-" * 72)

    print(
        "Available:",
        f"{data['mean_arterial_pressure'].notna().sum():,}",
    )

    print(
        "Median:",
        round(
            data["mean_arterial_pressure"].median(),
            2,
        ),
    )

    print(
        "Correlation with readmission:",
        round(
            data[
                "mean_arterial_pressure"
            ].corr(
                data[TARGET]
            ),
            4,
        ),
    )


# ============================================================
# MISSINGNESS VS READMISSION
# ============================================================

print()
print("=" * 72)
print("MISSINGNESS CHECK")
print("=" * 72)

for metric in available_metrics:

    present = data[
        data[metric].notna()
    ][TARGET]

    missing = data[
        data[metric].isna()
    ][TARGET]

    print()
    print(metric.upper())
    print("-" * 72)

    if len(present) > 0:

        print(
            "Readmission rate when available:",
            f"{present.mean() * 100:.2f}%",
        )

    if len(missing) > 0:

        print(
            "Readmission rate when missing:",
            f"{missing.mean() * 100:.2f}%",
        )

    else:

        print(
            "Readmission rate when missing: "
            "No missing observations"
        )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 72)
print("V4 METRIC CHECK COMPLETE")
print("=" * 72)