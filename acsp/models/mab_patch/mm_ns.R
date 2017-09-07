

# This is an auxiliary function, used by some of the pooling methods, which analyses the qrels and outputs
# the (binary) relevance of document docid
is_relevant <- function(qrels,docid)
{
  docid=toString(docid)
  fila = subset(qrels, DOC_ID==docid)
  if (nrow(fila)==0) {return(0)}
  else 
  {
    return(fila$REL)
  }
}

# input: 
#       - pool_folder: path of the folder of your computer where you have the runs. 
#                      the code assumes that all run filenames have the prefix "input"
#                      runs are assumed to follow the standard 6-column format
#       - qrels_path: path in your computer of the qrels file

# output: none (it just plots a graph with the avg rel found -accumulated-)


mm_ns_multiple_queries <- function(pool_folder, out_folder, qrels_path)
{
  # reads the qrel file into an R dataframe with appropriate column names
  qrels_df = read.table(qrels_path,header=FALSE)
  names(qrels_df) = c("QUERY","DUMMY","DOC_ID","REL")
  
  print(paste("Qrel file...",qrels_path,"...",nrow(qrels_df)," judgments."))
    
  # reads "input*" files from pool_folder and stores them into a list of data frames (run_rankings)
  files <- list.files(path=pool_folder, pattern = "input")
  print(paste("Processing...",pool_folder,"...",length(files)," run files",sep=""))
  

  run_rankings=list()

  for (f in files){
    filepath=paste(pool_folder,f,sep="/")  
    print(paste("file...",filepath))
    df = read.table(filepath,header=FALSE)
    names(df)=c("QUERY","LABEL","DOC_ID","RANK","SCORE","RUN")
    run_rankings[[length(run_rankings)+1]]=df
  } # files
  
  print(paste(length(run_rankings),"runs in the pool"))
  
    
  # now, we proceed query by query, and aggregate the statistics of relevant docs found at different number of judgments
  pool_depth=100
  
  queries= unique(qrels_df$QUERY)
  
  
  for (q in queries)
  {
  # this example produces the plot of pooling by DOCID.
  # just change this line to compute any other judgment sequence 
  # (by invoking any other pooling strategy from pooling_strategies.R) 
  judgments = pooling_mm(q, pool_depth, run_rankings, qrels_df, 0)
  
  # write file
  d=data.frame(doc = judgments)
  outpath=paste(out_folder, q, sep="/")  
  write.table(d, file = outpath, sep=",")

  # # data frame with the ranking of judgments and a chunk ID for each document
  # chunks=ceiling((1:length(judgments))/chunksize)
  # current_ranking=data.frame(DOCID=judgments, CHUNK=chunks, REL=rep(NA,length(judgments)))
  
  # # get the relevance assessments for the current query
  # current_qrels = subset(qrels_df, QUERY==q)
  
  # # assign the relevance column for each document in the sequence 
  # for (i in 1:length(judgments)) 
  # {
  #   current_ranking[i,"REL"]=is_relevant(current_qrels,current_ranking[i,"DOCID"])
  # }
  
  # print(paste("Query...",q,", pool size:", length(judgments), ". ", sum(current_ranking$REL)," docs are relevant.",sep="" ))
  
  } # for q in queries
  
  print("Finish")
}




  
  
  

  