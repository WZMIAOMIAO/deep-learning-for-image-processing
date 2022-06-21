from addict import Dict
from compression.engines.ie_engine import IEEngine
from compression.graph import load_model, save_model
from compression.graph.model_utils import compress_model_weights
from compression.pipeline.initializer import create_pipeline
from utils import MyDataLoader, Accuracy, read_split_data


def main():
    data_path = "/data/flower_photos"
    ir_model_xml = "ir_output/resnet34.xml"
    ir_model_bin = "ir_output/resnet34.bin"
    save_dir = "quant_ir_output"
    model_name = "quantized_resnet34"
    img_w = 224
    img_h = 224

    model_config = Dict({
        'model_name': 'resnet34',
        'model': ir_model_xml,
        'weights': ir_model_bin
    })
    engine_config = Dict({
        'device': 'CPU',
        'stat_requests_number': 2,
        'eval_requests_number': 2
    })
    dataset_config = {
        'data_source': data_path
    }
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

    # Steps 1-7: Model optimization
    # Step 1: Load the model.
    model = load_model(model_config)

    # Step 2: Initialize the data loader.
    _, _, val_images_path, val_images_label = read_split_data(data_path, val_rate=0.2)
    data_loader = MyDataLoader(dataset_config, val_images_path, val_images_label, img_w, img_h)

    # Step 3 (Optional. Required for AccuracyAwareQuantization): Initialize the metric.
    metric = Accuracy(top_k=1)

    # Step 4: Initialize the engine for metric calculation and statistics collection.
    engine = IEEngine(engine_config, data_loader, metric)

    # Step 5: Create a pipeline of compression algorithms.
    pipeline = create_pipeline(algorithms, engine)

    # Step 6: Execute the pipeline.
    compressed_model = pipeline.run(model)

    # Step 7 (Optional): Compress model weights quantized precision
    #                    in order to reduce the size of final .bin file.
    compress_model_weights(compressed_model)

    # Step 8: Save the compressed model to the desired path.
    compressed_model_paths = save_model(model=compressed_model,
                                        save_path=save_dir,
                                        model_name=model_name)

    # Step 9: Compare accuracy of the original and quantized models.
    metric_results = pipeline.evaluate(model)
    if metric_results:
        for name, value in metric_results.items():
            print(f"Accuracy of the original model: {name}: {value}")

    metric_results = pipeline.evaluate(compressed_model)
    if metric_results:
        for name, value in metric_results.items():
            print(f"Accuracy of the optimized model: {name}: {value}")


if __name__ == '__main__':
    main()
