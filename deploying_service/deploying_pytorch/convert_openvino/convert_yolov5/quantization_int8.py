import time
from addict import Dict
from compression.engines.ie_engine import IEEngine
from compression.graph import load_model, save_model
from compression.graph.model_utils import compress_model_weights
from compression.pipeline.initializer import create_pipeline
from yaspin import yaspin
from utils import MyDataLoader, MAPMetric


def main():
    data_path = "/data/coco2017"
    ir_model_xml = "ir_output/yolov5s.xml"
    ir_model_bin = "ir_output/yolov5s.bin"
    save_dir = "quant_ir_output"
    model_name = "quantized_yolov5s"
    img_w = 640
    img_h = 640

    model_config = Dict({
        'model_name': 'yolov5s',
        'model': ir_model_xml,
        'weights': ir_model_bin,
        'inputs': 'images',
        'outputs': 'output'
    })
    engine_config = Dict({'device': 'CPU'})

    algorithms = [
        {
            'name': 'DefaultQuantization',
            'params': {
                'target_device': 'CPU',
                'preset': 'performance',
                'stat_subset_size': 300
            }
        }
    ]

    # Step 1: Load the model.
    model = load_model(model_config)

    # Step 2: Initialize the data loader.
    data_loader = MyDataLoader(data_path, "val", (img_h, img_w))

    # Step 3: initialize the metric
    # For DefaultQuantization, specifying a metric is optional: metric can be set to None
    metric = MAPMetric(map_value="map")

    # Step 4: Initialize the engine for metric calculation and statistics collection.
    engine = IEEngine(config=engine_config, data_loader=data_loader, metric=metric)

    # Step 5: Create a pipeline of compression algorithms.
    pipeline = create_pipeline(algorithms, engine)

    # Step 6: Execute the pipeline to quantize the model
    algorithm_name = pipeline.algo_seq[0].name
    with yaspin(
            text=f"Executing POT pipeline on {model_config['model']} with {algorithm_name}"
    ) as sp:
        start_time = time.perf_counter()
        compressed_model = pipeline.run(model)
        end_time = time.perf_counter()
        sp.ok("✔")
    print(f"Quantization finished in {end_time - start_time:.2f} seconds")

    # Step 7 (Optional): Compress model weights to quantized precision
    #                    in order to reduce the size of the final .bin file
    compress_model_weights(compressed_model)

    # Step 8: Save the compressed model to the desired path.
    # Set save_path to the directory where the compressed model should be stored
    compressed_model_paths = save_model(
        model=compressed_model,
        save_path=save_dir,
        model_name=model_name,
    )

    compressed_model_path = compressed_model_paths[0]["model"]
    print("The quantized model is stored at", compressed_model_path)

    # Compute the mAP on the quantized model and compare with the mAP on the FP16 IR model.
    ir_model = load_model(model_config=model_config)
    evaluation_pipeline = create_pipeline(algo_config=dict(), engine=engine)

    with yaspin(text="Evaluating original IR model") as sp:
        original_metric = evaluation_pipeline.evaluate(ir_model)

    if original_metric:
        for key, value in original_metric.items():
            print(f"The {key} score of the original model is {value:.5f}")

    with yaspin(text="Evaluating quantized IR model") as sp:
        quantized_metric = pipeline.evaluate(compressed_model)

    if quantized_metric:
        for key, value in quantized_metric.items():
            print(f"The {key} score of the quantized INT8 model is {value:.5f}")


if __name__ == '__main__':
    main()
