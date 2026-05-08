python3 tools/profile_cluster_gcn_spmm.py \
  --datasets ogbn_arxiv,ogbn_products,flickr,yelp,amazon_products,reddit \
  --epochs 3 \
  --cluster-parts 0 \
  --cluster-batch-size 20 \
  --warmup-steps 5 \
  --output-dir results/gnn_final/cluster_gcn_spmm_six_full


python3 tools/profile_graphsage_sampled_spmm.py \
  --datasets ogbn_arxiv,ogbn_products,flickr,yelp,amazon_products,reddit \
  --epochs 3 \
  --batch-size 1024 \
  --fanout 10,10 \
  --warmup-steps 5 \
  --output-dir results/gnn_final/graphsage_spmm_six_full
