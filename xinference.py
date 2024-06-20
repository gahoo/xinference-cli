import argparse
from xinference_client import RESTfulClient as Client

def launch_model(args):
    # 初始化客户端
    client = Client(args.base_url)
    
    # 启动模型
    model_uid = client.launch_model(
        model_name=args.model_name,
        model_type=args.model_type,
        model_engine=args.model_engine,
        model_uid=args.model_uid,
        model_size_in_billions=args.model_size_in_billions,
        model_format=args.model_format,
        quantization=args.quantization,
        replica=args.replica,
        n_gpu=args.n_gpu,
        peft_model_config=args.peft_model_config,
        request_limits=args.request_limits,
        worker_ip=args.worker_ip,
        gpu_idx=args.gpu_idx
    )
    
    # 输出模型 UID
    print(f"Model launched with UID: {model_uid}")

def main():
    parser = argparse.ArgumentParser(description="命令行工具用于启动模型")
    
    subparsers = parser.add_subparsers(help="子命令", dest="command")
    
    # 创建 launch 子命令
    launch_parser = subparsers.add_parser("launch", help="启动模型")
    launch_parser.add_argument("--base_url", type=str, required=True, help="服务器的基础 URL")
    launch_parser.add_argument("--model_name", type=str, required=True, help="模型名称")
    launch_parser.add_argument("--model_type", type=str, default='LLM', help="模型类型")
    launch_parser.add_argument("--model_engine", type=str, default=None, help="模型引擎")
    launch_parser.add_argument("--model_uid", type=str, default=None, help="模型 UID")
    launch_parser.add_argument("--model_size_in_billions", type=float, default=None, help="模型大小（以十亿为单位）")
    launch_parser.add_argument("--model_format", type=str, default=None, help="模型格式")
    launch_parser.add_argument("--quantization", type=str, default=None, help="模型量化方式")
    launch_parser.add_argument("--replica", type=int, default=1, help="副本数量")
    launch_parser.add_argument("--n_gpu", type=str, default='auto', help="使用的 GPU 数量")
    launch_parser.add_argument("--peft_model_config", type=dict, default=None, help="PEFT 模型配置")
    launch_parser.add_argument("--request_limits", type=int, default=None, help="请求限制")
    launch_parser.add_argument("--worker_ip", type=str, default=None, help="工作节点 IP")
    launch_parser.add_argument("--gpu_idx", type=int, nargs='+', default=None, help="GPU 索引")
    
    args = parser.parse_args()
    
    if args.command == "launch":
        launch_model(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
