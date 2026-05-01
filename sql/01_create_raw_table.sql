CREATE EXTERNAL TABLE IF NOT EXISTS wa_property_db.wa_property_raw (
  address string,
  suburb string,
  price double,
  bedrooms int,
  bathrooms int,
  garage int,
  land_area double,
  floor_area double,
  build_year int,
  cbd_dist double,
  nearest_stn string,
  nearest_stn_dist double,
  date_sold string,
  postcode int,
  latitude double,
  longitude double,
  nearest_sch string,
  nearest_sch_dist double,
  nearest_sch_rank double
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
  'separatorChar' = ',',
  'quoteChar' = '"',
  'escapeChar' = '\\'
)
LOCATION 's3://personal-wa-property-storage-337164669284-ap-southeast-2-an/raw/property/'
TBLPROPERTIES (
  'skip.header.line.count'='1'
);