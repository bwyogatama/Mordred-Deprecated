DROP TABLE if exists lineorder;
DROP TABLE if exists ddate;
DROP TABLE if exists part;
DROP TABLE if exists supplier;
DROP TABLE if exists customer;

CREATE  TABLE customer (
  c_custkey     INTEGER,
  c_name        TEXT ENCODING DICT,
  c_address     TEXT ENCODING DICT,
  c_city        INTEGER,
  c_nation      INTEGER,
  c_region      INTEGER,
  c_phone       TEXT ENCODING DICT,
  c_mktsegment  TEXT ENCODING DICT,
  Dummy         TEXT ENCODING NONE
);

CREATE  TABLE ddate (
  d_datekey           INTEGER,
  d_date              TEXT ENCODING DICT,
  d_dayofweek         TEXT ENCODING DICT,
  d_month             TEXT ENCODING DICT,
  d_year              INTEGER ,
  d_yearmonthnum      INTEGER ,
  d_yearmonth         TEXT ENCODING DICT,
  d_daynuminweek      INTEGER ,
  d_daynuminmonth     INTEGER ,
  d_daynuminyear      INTEGER ,
  d_monthnuminyear    INTEGER ,
  d_weeknuminyear     INTEGER ,
  d_sellingseason     TEXT ENCODING DICT,
  d_lastdayinweekfl   TEXT ENCODING DICT,
  d_lastdayinmonthfl  TEXT ENCODING DICT,
  d_holidayfl         TEXT ENCODING DICT,
  d_weekdayfl         TEXT ENCODING DICT,
  Dummy               TEXT ENCODING NONE
);

CREATE  TABLE  part (
  p_partkey    INTEGER,
  p_name       TEXT ENCODING DICT,
  p_mfgr       INTEGER,
  p_category   INTEGER,
  p_brand1     INTEGER,
  p_color      TEXT ENCODING DICT,
  p_type       TEXT ENCODING DICT,
  p_size       INTEGER,
  p_container  TEXT ENCODING DICT,
  Dummy        TEXT ENCODING NONE
);

CREATE  TABLE  supplier (
  s_suppkey  INTEGER,
  s_name     TEXT ENCODING DICT,
  s_address  TEXT ENCODING DICT,
  s_city     INTEGER,
  s_nation   INTEGER,
  s_region   INTEGER,
  s_phone    TEXT ENCODING DICT,
  Dummy      TEXT ENCODING NONE
);

CREATE  TABLE  lineorder (
  lo_orderkey       BIGINT ,
  lo_linenumber     INTEGER ,
  lo_custkey        INTEGER ,
  lo_partkey        INTEGER ,
  lo_suppkey        INTEGER ,
  lo_orderdate      INTEGER ,    
  lo_orderpriority  TEXT ENCODING DICT,
  lo_shippriority   TEXT ENCODING DICT,
  lo_quantity       INTEGER ,
  lo_extendedprice  INTEGER ,
  lo_ordtotalprice  INTEGER ,
  lo_discount       INTEGER ,
  lo_revenue        INTEGER ,
  lo_supplycost     INTEGER ,
  lo_tax            INTEGER ,
  lo_commitdate     INTEGER ,
  lo_shipmod        TEXT ENCODING DICT,
  Dummy             TEXT ENCODING NONE
);