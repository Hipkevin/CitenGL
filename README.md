# Heterogeneous Hypergraph Learning for Literature Retrieval Based on Citation Intents

Literature retrieval helps scientists find previous work that is relative to their own research or even get new research ideas. However, the discrepancy between retrieval results and the ultimate intention of citation is neglected by most literature retrieval models. Citation intent refers to the researcher’s motivation for citing a paper. A citation intent graph with homogeneous nodes and
heterogeneous hyperedges can represent different types of citation intents. By leveraging the citation intent information included in a hypergraph, a retrieval model can guide researchers on where to cite its retrieval result by understanding the citation behaviour in the graph. We present a ranking model called
CitenGL (Citation Intent Graph Learning) that aims to extract citation intent information and textual matching signals. The proposed model consists of a heterogeneous hypergraph encoder and a lightweight deep fusion unit for efficiency
trade-offs. Compared to traditional literature retrieval, our model fills the gap between retrieval results and citation intention and yields an understandable graph-structured output. We evaluated our model on publicly available full-text paper datasets. Experimental results show that CitenGL outperforms most
existing neural ranking models that only consider textual information, which illustrates the effectiveness of integrating citation intent information with textual information. Further ablation analyses show how citation intent information
complements text-matching signals and citation networks.

## Project Main Structure
    -util
        dataTool  # graph data api
        graph_model
        textEncoderTool  # graph + text encoder (train text embedding)

    run_*  # main file

