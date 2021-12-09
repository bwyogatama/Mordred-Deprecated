DROP TABLE if exists lineorder;
DROP TABLE if exists dates;
DROP TABLE if exists part;
DROP TABLE if exists supplier;
DROP TABLE if exists customer;

CREATE TABLE customer (
  c_custkey     INTEGER,
  c_name        TEXT ENCODING DICT,
  c_address     TEXT ENCODING DICT,
  c_city        TEXT ENCODING DICT,
  c_nation      TEXT ENCODING DICT,
  c_region      TEXT ENCODING DICT,
  c_phone       TEXT ENCODING DICT,
  c_mktsegment  TEXT ENCODING DICT,
  Dummy         TEXT ENCODING NONE
) with (fragment_size = 1000000);

CREATE  TABLE dates (
  d_datekey           DATE,
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
) with (fragment_size = 1000000);

CREATE  TABLE  part (
  p_partkey    INTEGER,
  p_name       TEXT ENCODING DICT,
  p_mfgr       TEXT ENCODING DICT,
  p_category   TEXT ENCODING DICT,
  p_brand1     TEXT ENCODING DICT,
  p_color      TEXT ENCODING DICT,
  p_type       TEXT ENCODING DICT,
  p_size       INTEGER ,
  p_container  TEXT ENCODING DICT,
  Dummy        TEXT ENCODING NONE
) with (fragment_size = 1000000);

CREATE  TABLE  supplier (
  s_suppkey INTEGER,
  s_name     TEXT ENCODING DICT,
  s_address  TEXT ENCODING DICT,
  s_city     TEXT ENCODING DICT,
  s_nation   TEXT ENCODING DICT,
  s_region   TEXT ENCODING DICT,
  s_phone    TEXT ENCODING DICT,
  Dummy      TEXT ENCODING NONE
) with (fragment_size = 1000000);

CREATE  TABLE  lineorder (
  lo_orderkey       BIGINT ,
  lo_linenumber     INTEGER ,
  lo_custkey        INTEGER ,
  lo_partkey        INTEGER ,
  lo_suppkey        INTEGER ,
  lo_orderdate      DATE ,    
  lo_orderpriority  TEXT ENCODING DICT,
  lo_shippriority   TEXT ENCODING DICT,
  lo_quantity       INTEGER ,
  lo_extendedprice  INTEGER ,
  lo_ordtotalprice  INTEGER ,
  lo_discount       INTEGER ,
  lo_revenue        INTEGER ,
  lo_supplycost     INTEGER ,
  lo_tax            INTEGER ,
  lo_commitdate     DATE ,
  lo_shipmod        TEXT ENCODING DICT,
  Dummy             TEXT ENCODING NONE
) with (fragment_size = 1000000);