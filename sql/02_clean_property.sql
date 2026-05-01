USE wa_property_db;

CREATE TABLE wa_property_clean
WITH (
  format = 'PARQUET',
  external_location = 's3://personal-wa-property-storage-337164669284-ap-southeast-2-an/processed/property_clean/'
) AS
SELECT *
FROM wa_property_raw
WHERE bedrooms <= 5
  AND bathrooms <= 3
  AND garage <= 2;