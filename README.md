# ApiRetrieval
# Prepare
```
pip install -r requirements.txt
conda install -c conda-forge faiss-gpu
```
# Results
## nvidia/NV-Embed-v2
A800 80G

| Test Size | k | Overall Accuracy (No Dedup) | Overall Accuracy (Dedup) | Overall Precision |
|-----------|---|-----------------------------|--------------------------|-------------------|
| 0.25      | 1 | 0.9259                      | 0.9259                   | N/A               |
| 0.25      | 3 | 0.9852                      | 0.9852                   | 0.9012            |
| 0.25      | 5 | 1.0000                      | 1.0000                   | 0.8815            |
| 0.5       | 1 | 0.8859                      | 0.8859                   | N/A               |
| 0.5       | 3 | 0.9696                      | 0.9696                   | 0.8758            |
| 0.5       | 5 | 0.9848                      | 0.9848                   | 0.8540            |
| 0.75      | 1 | 0.8824                      | 0.8824                   | N/A               |
| 0.75      | 3 | 0.9719                      | 0.9719                   | 0.8380            |
| 0.75      | 5 | 0.9949                      | 0.9949                   | 0.7908            |


| k | Test Size | Overall Accuracy (No Dedup) | Overall Accuracy (Dedup) | Overall Precision |
|---|-----------|-----------------------------|--------------------------|-------------------|
| 1 | 0.25      | 0.9259                      | 0.9259                   | N/A               |
| 1 | 0.5       | 0.8859                      | 0.8859                   | N/A               |
| 1 | 0.75      | 0.8824                      | 0.8824                   | N/A               |
| 3 | 0.25      | 0.9852                      | 0.9852                   | 0.9012            |
| 3 | 0.5       | 0.9696                      | 0.9696                   | 0.8758            |
| 3 | 0.75      | 0.9719                      | 0.9719                   | 0.8380            |
| 5 | 0.25      | 1.0000                      | 1.0000                   | 0.8815            |
| 5 | 0.5       | 0.9848                      | 0.9848                   | 0.8540            |
| 5 | 0.75      | 0.9949                      | 0.9949                   | 0.7908            |

注：上述两个表仅排列方式不同，数据完全相同。

## BAAI/bge-en-icl
A800 80G

| Test Size | k  | Overall Accuracy No Dedup | Overall Accuracy Dedup | Overall Precision |
|-----------|----|---------------------------|------------------------|-------------------|
| 0.25      | 1  | 0.8667                    | 0.8667                 | N/A               |
| 0.25      | 3  | 0.9630                    | 0.9630                 | 0.8790            |
| 0.25      | 5  | 0.9778                    | 0.9778                 | 0.8533            |
| 0.5       | 1  | 0.8707                    | 0.8707                 | N/A               |
| 0.5       | 3  | 0.9544                    | 0.9544                 | 0.8302            |
| 0.5       | 5  | 0.9696                    | 0.9696                 | 0.8015            |
| 0.75      | 1  | 0.8414                    | 0.8414                 | N/A               |
| 0.75      | 3  | 0.9616                    | 0.9616                 | 0.7954            |
| 0.75      | 5  | 0.9923                    | 0.9923                 | 0.7284            |

| k  | Test Size | Overall Accuracy No Dedup | Overall Accuracy Dedup | Overall Precision |
|----|-----------|---------------------------|------------------------|-------------------|
| 1  | 0.25      | 0.8667                    | 0.8667                 | N/A               |
| 1  | 0.5       | 0.8707                    | 0.8707                 | N/A               |
| 1  | 0.75      | 0.8414                    | 0.8414                 | N/A               |
| 3  | 0.25      | 0.9630                    | 0.9630                 | 0.8790            |
| 3  | 0.5       | 0.9544                    | 0.9544                 | 0.8302            |
| 3  | 0.75      | 0.9616                    | 0.9616                 | 0.7954            |
| 5  | 0.25      | 0.9778                    | 0.9778                 | 0.8533            |
| 5  | 0.5       | 0.9696                    | 0.9696                 | 0.8015            |
| 5  | 0.75      | 0.9923                    | 0.9923                 | 0.7284            |

注：上述两个表仅排列方式不同，数据完全相同。

## dunzhang/stella_en_1.5B_v5
RTX4090 24G

| Test Size | k   | Overall Accuracy (No Dedup) | Overall Accuracy (Dedup) | Overall Precision |
|-----------|-----|-----------------------------|--------------------------|-------------------|
| 0.25      | 1   | 0.8889                      | 0.8889                   | N/A               |
| 0.25      | 3   | 0.9852                      | 0.9852                   | 0.9012            |
| 0.25      | 5   | 1.0000                      | 1.0000                   | 0.8948            |
| 0.5       | 1   | 0.9087                      | 0.9087                   | N/A               |
| 0.5       | 3   | 0.9848                      | 0.9848                   | 0.8821            |
| 0.5       | 5   | 0.9924                      | 0.9924                   | 0.8487            |
| 0.75      | 1   | 0.8645                      | 0.8645                   | N/A               |
| 0.75      | 3   | 0.9693                      | 0.9693                   | 0.8167            |
| 0.75      | 5   | 0.9821                      | 0.9821                   | 0.7524            |

| k   | Test Size | Overall Accuracy (No Dedup) | Overall Accuracy (Dedup) | Overall Precision |
|-----|-----------|-----------------------------|--------------------------|-------------------|
| 1   | 0.25      | 0.8889                      | 0.8889                   | N/A               |
| 1   | 0.5       | 0.9087                      | 0.9087                   | N/A               |
| 1   | 0.75      | 0.8645                      | 0.8645                   | N/A               |
| 3   | 0.25      | 0.9852                      | 0.9852                   | 0.9012            |
| 3   | 0.5       | 0.9848                      | 0.9848                   | 0.8821            |
| 3   | 0.75      | 0.9693                      | 0.9693                   | 0.8167            |
| 5   | 0.25      | 1.0000                      | 1.0000                   | 0.8948            |
| 5   | 0.5       | 0.9924                      | 0.9924                   | 0.8487            |
| 5   | 0.75      | 0.9821                      | 0.9821                   | 0.7524            |

注：上述两个表仅排列方式不同，数据完全相同。
