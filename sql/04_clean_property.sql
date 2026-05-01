CREATE TABLE wa_property_db.{table_name}
WITH (
  format = 'PARQUET',
  external_location = '{output_path}'
) AS
SELECT
  address,
  suburb,
  price,
  bedrooms,
  bathrooms,
  garage,
  land_area,
  floor_area,
  cbd_dist,
  nearest_stn,
  nearest_stn_dist,
  date_sold,
  postcode,
  latitude,
  longitude,
  nearest_sch,
  nearest_sch_dist,
  nearest_sch_rank
FROM wa_property_db.wa_property_raw
WHERE bedrooms <= 5
  AND bathrooms <= 3
  AND garage <= 2
  AND land_area BETWEEN 400 AND 800
  AND date_parse(date_sold, '%m-%Y') >= DATE '2015-01-01'

ORDER BY postcode;