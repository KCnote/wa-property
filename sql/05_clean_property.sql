CREATE TABLE wa_property_db.wa_property_clean
WITH (
  format = 'PARQUET',
  external_location = '{output_path}'
) AS
SELECT *
FROM wa_property_db.wa_property_raw
WHERE bedrooms <= 5
  AND bathrooms <= 3
  AND garage <= 2;