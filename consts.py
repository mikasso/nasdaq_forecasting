SYMBOLS = ["AAPL", "MSFT", "GOOGL", "META", "ORCL"]
DATA_PATH = "data/nasdaq/parquet"
MERGED_PATH = "data/nasdaq/merged"
DROP_COLUMNS = [
    "seqnum",
    "mktcenter",
    "salescondition",
    "canceled",
    "dottchar",
    "issuechar",
    "msgseqnum",
    "originalmsgseqnum",
    "submkt",
]
START_DATE = "20080101"
