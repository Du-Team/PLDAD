# PLDAD: Progressive Latent Diffusion-Driven Anomaly Detection for Multivariate Time Series
## Requirements
 * PyTorch 2.5.1
 * CUDA 10.1 (to allow use of GPU, not compulsory)

# Dataset

* SMAP and MSL:

```
wget https://s3-us-west-2.amazonaws.com/telemanom/data.zip && unzip data.zip && rm data.zip

cd data && wget https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv
```

* SMD:

```
https://github.com/NetManAIOps/OmniAnomaly
```

* SWaT:

```
http://itrust.sutd.edu.sg/research/dataset
```


* Run the code

```
./run.sh
```

where `<dataset>` is one of `SMAP`, `MSL`, `SMD`, `SWAT`, `PSM`, `ASD`

For more related researches, please visit my homepage: https://dumingjing.github.io/. For data and discussion, please message Mingjing Du (杜明晶@江苏师范大学): dumj@jsnu.edu.cn.

