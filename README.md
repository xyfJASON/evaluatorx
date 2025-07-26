<h1 align="center">evaluatorx</h1>

Implements evaluators based on torch and torchvision.

Currently supports:

<table>
  <tr>
    <th>Task</th>
    <th>Evaluator Class</th>
    <th>Metrics</th>
  </tr>
  <tr>
    <td>Affine-invariant Depth Estimation</td>
    <td><code>DepthEvaluator</code></td>
    <td>absrel, rmse, delta1, delta2, delta3</td>
  </tr>
  <tr>
    <td>Surface Normal Estimation</td>
    <td><code>NormalEvaluator</code></td>
    <td>mean, median, rmse, a1, a2, a3, a4, a5</td>
  </tr>
  <tr>
    <td>Semantic Segmentation</td>
    <td><code>SegmentationEvaluator</code></td>
    <td>miou, fwiou, class-iou, macc, pacc, class-acc</td>
  </tr>
</table>



## Installation

```shell
pip install git+https://github.com/xyfJASON/evaluatorx.git
```
