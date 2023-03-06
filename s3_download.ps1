$SYMBOL=$args[0]
$OUTPUT_DIR = "./data/parquet/${SYMBOL}/"

Remove-Item $OUTPUT_DIR -Recurse -ErrorAction Ignore
mkdir  $OUTPUT_DIR

aws s3 ls "s3://ncpgisahub-pro-dod-etl/trades/symbol=${SYMBOL}/" --recursive --human-readable --summarize > "./data/meta/" ${SYMBOL}_summarize.txt
aws s3 cp "s3://ncpgisahub-pro-dod-etl/trades/symbol=${SYMBOL}/" $OUTPUT_DIR --recursive
