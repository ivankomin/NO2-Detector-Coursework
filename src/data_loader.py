import ee
import pandas as pd


def initialize_gee(project_name="ml-cv-class-2026-496115"):
    try:
        if project_name:
            ee.Initialize(project=project_name)
        else:
            ee.Initialize()
        print("GEE успішно ініціалізовано.")
    except Exception as e:
        print(f"Помилка ініціалізації GEE: {e}")


def _fetch_collection_time_series(
    collection_id: str,
    bands: list,
    roi: ee.Geometry,
    start_date: str,
    end_date: str,
    scale: int,
) -> pd.DataFrame:
    coll = ee.ImageCollection(collection_id).filterBounds(roi).select(bands)

    start = ee.Date(start_date)
    end = ee.Date(end_date)
    n_days = end.difference(start, "days")
    # create a list of day offsets from the start date
    day_list = ee.List.sequence(0, n_days.subtract(1))

    def process_day(day_offset):
        # create a date from the day offset and filter the collection to that day
        date = start.advance(day_offset, "days")
        daily_coll = coll.filterDate(date, date.advance(1, "days"))

        # calculate the mean for that day and return it as a feature with the date as a property
        mean_obj = daily_coll.mean().reduceRegion(
            reducer=ee.Reducer.mean(), geometry=roi, scale=scale, maxPixels=1e9
        )
        return ee.Feature(None, mean_obj.set("date", date.format("yyyy-MM-dd")))

    # fetch the features for all days and load them into a pandas DataFrame
    features = ee.FeatureCollection(day_list.map(process_day)).getInfo()["features"]

    records = []
    for f in features:
        props = f["properties"]
        # ignore records where all bands are None (e.g. due to cloud cover or no data)
        if props and any(props.get(b) is not None for b in bands):
            records.append(props)

    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna().set_index("date").sort_index()
        df = df.groupby("date").mean()
    return df


def get_comprehensive_data(
    lon: float, lat: float, buffer_km: float, start_date: str, end_date: str
) -> pd.DataFrame:
    point = ee.Geometry.Point([lon, lat])
    roi = point.buffer(buffer_km * 1000)

    print("   -> Збір NO2 та хмарності...")
    df_no2 = _fetch_collection_time_series(
        "COPERNICUS/S5P/OFFL/L3_NO2",
        ["NO2_column_number_density", "cloud_fraction"],
        roi,
        start_date,
        end_date,
        3500,
    )

    print("   -> Збір CO, SO2 та Аерозолів...")
    df_co = _fetch_collection_time_series(
        "COPERNICUS/S5P/OFFL/L3_CO",
        ["CO_column_number_density"],
        roi,
        start_date,
        end_date,
        3500,
    )
    df_so2 = _fetch_collection_time_series(
        "COPERNICUS/S5P/OFFL/L3_SO2",
        ["SO2_column_number_density"],
        roi,
        start_date,
        end_date,
        3500,
    )
    df_aer = _fetch_collection_time_series(
        "COPERNICUS/S5P/OFFL/L3_AER_AI",
        ["absorbing_aerosol_index"],
        roi,
        start_date,
        end_date,
        3500,
    )

    print("   -> Збір метеорології (ERA5-Land)...")
    meteo_bands = [
        "temperature_2m",
        "dewpoint_temperature_2m",
        "u_component_of_wind_10m",
        "v_component_of_wind_10m",
    ]
    df_meteo = _fetch_collection_time_series(
        "ECMWF/ERA5_LAND/DAILY_AGGR", meteo_bands, roi, start_date, end_date, 10000
    )

    print("   -> Збір висоти прикордонного шару PBLH (ERA5 Hourly)...")
    df_pblh = _fetch_collection_time_series(
        "ECMWF/ERA5/HOURLY", ["boundary_layer_height"], roi, start_date, end_date, 10000
    )

    df_merged = df_no2.join([df_co, df_so2, df_aer, df_meteo, df_pblh], how="inner")
    df_merged = df_merged[df_merged["cloud_fraction"] <= 0.3]

    return df_merged
