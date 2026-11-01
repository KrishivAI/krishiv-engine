import sys,os,glob
from pyspark.sql import SparkSession
out=sys.argv[1]
spark=SparkSession.builder.master('local[2]').config('spark.sql.catalogImplementation','in-memory').config('spark.ui.enabled','false').getOrCreate()
df=spark.createDataFrame([(1,'a,b','x"y'),(2,None,'multi\nline'),(3,'é','')],'id int, s string, t string')
for fmt in ['csv','json','text']:
    d = df.select('s') if fmt=='text' else df
    for c in ['none','gzip','deflate','bzip2','lz4','snappy','zstd','zstandard','uncompressed','GZIP']:
        try:
            d.coalesce(1).write.mode('overwrite').option('compression',c).format(fmt).save(f'{out}/{fmt}-{c}')
            print(fmt,c,'OK',sorted(os.listdir(f'{out}/{fmt}-{c}')))
        except Exception as e:
            print(fmt,c,'ERR',str(e).splitlines()[0][:200])
spark.stop()
