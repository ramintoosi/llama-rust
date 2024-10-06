#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use anyhow::{anyhow, bail, Context, Result};
use clap::{arg, Parser, Subcommand};
use hf_hub::api::sync::ApiBuilder;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::ggml_time_us;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::kv_overrides::ParamOverrideValue;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::{AddBos, Special};
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use std::ffi::CString;
use std::io::Write;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::pin::pin;
use std::str::FromStr;
use std::time::Duration;
use log::error;

#[derive(Subcommand, Debug, Clone)]
enum Model {
    /// Use an already downloaded model
    #[clap(name = "local")]
    Local {
        /// The path to the model. e.g. `../hub/models--TheBloke--Llama-2-7B-Chat-GGUF/blobs/08a5566d61d7cb6b420c3e4387a39e0078e1f2fe5f055f3a03887385304d4bfa`
        /// or `./llama-3.2-1b-instruct-q8_0.gguf`
        path: PathBuf,
    },
    /// Download a model from huggingface (or use a cached version)
    #[clap(name = "hf-model")]
    HuggingFace {
        /// the repo containing the model. e.g. `TheBloke/Llama-2-7B-Chat-GGUF`
        repo: String,
        /// the model name. e.g. `llama-2-7b-chat.Q4_K_M.gguf`
        model: String,
    },
}

impl Model {
    /// Convert the model to a path - may download from huggingface
    fn get_or_load(self) -> Result<PathBuf> {
        match self {
            Model::Local { path } => Ok(path),
            Model::HuggingFace { model, repo } => ApiBuilder::new()
                .with_progress(true)
                .build()
                .with_context(|| "unable to create huggingface api")?
                .model(repo)
                .get(&model)
                .with_context(|| "unable to download model"),
        }
    }
}


#[derive(clap::ValueEnum, Clone, Debug)]
enum Mode {
    Chat,
    Completion,
}

#[derive(Parser, Debug, Clone)]
struct Args {
    /// The path to the model
    #[command(subcommand)]
    model: Model,

    /// The mode of the code: completion or chat
    #[clap(value_enum, short = 'm', long, default_value = "chat")]
    mode: Mode,

    // /// The prompt to use - valid only if the mode is `completion`
    // #[clap(short = 'p', long, required_if_eq("mode", "completion"))]
    // prompt: Option<String>,

    /// set the length of the prompt + output in tokens
    #[clap(long, default_value_t = 512)]
    max_token: i32,

    /// override some parameters of the model
    #[clap(short = 'o', value_parser = parse_key_val)]
    key_value_overrides: Vec<(String, ParamOverrideValue)>,

    /// how many layers to keep on the gpu - zero is cpu mode
    #[clap(
        short = 'g',
        long,
        help = "how many layers to keep on the gpu - zero is cpu mode (default: 0)"
    )]
    n_gpu_layers: u32,

    /// set the seed for the RNG
    #[clap(short = 's', long, default_value_t=561371, help = "RNG seed (default: 1234)")]
    seed: u32,

    /// number of threads to use during generation
    #[clap(
        long,
        help = "number of threads to use during generation (default: use all available threads)"
    )]
    threads: Option<i32>,
    #[clap(
        long,
        help = "number of threads to use during batch and prompt processing (default: use all available threads)"
    )]
    threads_batch: Option<i32>,

    /// size of the prompt context
    #[clap(
        short = 'c',
        long,
        help = "size of the prompt context (default: loaded from the model)"
    )]
    ctx_size: Option<NonZeroU32>,

}

/// Parse a single key-value pair
fn parse_key_val(s: &str) -> Result<(String, ParamOverrideValue)> {
    let pos = s
        .find('=')
        .ok_or_else(|| anyhow!("invalid KEY=value: no `=` found in `{}`", s))?;
    let key = s[..pos].parse()?;
    let value: String = s[pos + 1..].parse()?;
    let value = i64::from_str(&value)
        .map(ParamOverrideValue::Int)
        .or_else(|_| f64::from_str(&value).map(ParamOverrideValue::Float))
        .or_else(|_| bool::from_str(&value).map(ParamOverrideValue::Bool))
        .map_err(|_| anyhow!("must be one of i64, f64, or bool"))?;

    Ok((key, value))
}

fn main() {

    let args: Args = Args::parse();

    // init LLM
    let backend = LlamaBackend::init()
        .expect("Could not initialize Llama backend");

    // offload all layers to the gpu
    let model_params = {
        if args.n_gpu_layers > 0 {
            LlamaModelParams::default().with_n_gpu_layers(args.n_gpu_layers)
        } else {
            LlamaModelParams::default()
        }
    };

    let mut model_params = pin!(model_params);
    
    for (k, v) in &args.key_value_overrides {
        let k = CString::new(k.as_bytes()).with_context(|| format!("invalid key: {k}")).unwrap();
        model_params.as_mut().append_kv_override(k.as_c_str(), *v);
    }

    let model_path = args.model
        .get_or_load()
        .with_context(|| "failed to get model from args").unwrap();

    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
        .with_context(|| "unable to load model").unwrap();


    // initialize the context
    let mut ctx_params = LlamaContextParams::default()
        .with_n_ctx(args.ctx_size.or(Some(NonZeroU32::new(2048).unwrap())))
        .with_seed(args.seed);
    if let Some(threads) = args.threads {
        ctx_params = ctx_params.with_n_threads(threads);
    }
    if let Some(threads_batch) = args.threads_batch.or(args.threads) {
        ctx_params = ctx_params.with_n_threads_batch(threads_batch);
    }

    let mut ctx = model
        .new_context(&backend, ctx_params)
        .with_context(|| "unable to create the llama_context").unwrap();
    
    // main loop
    let mut input_to_model = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful assistant<|eot_id|>".to_string();
    let mut input = String::new();
    
    print!("Assistant: How can I help you today?\n");
    loop {
        
        input.clear();
        println!("\nYou: ");
        std::io::stdout().flush().unwrap();
        std::io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();
        
        if input.is_empty() {
            break;
        }
        
        
        let input_formatted = format!("<|start_header_id|>user<|end_header_id|>{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>", input);
        
        input_to_model.push_str(input_formatted.as_str());
        
        

        // tokenize the prompt
        let tokens_list = model
            .str_to_token(&input_to_model, AddBos::Never)
            .with_context(|| format!("failed to tokenize {input_to_model}")).unwrap();

        let n_cxt = ctx.n_ctx() as i32;
        let n_kv_req = tokens_list.len() as i32 + (args.max_token - tokens_list.len() as i32);
        // let max_token = args.max_token;

        // eprintln!("n_len = {max_token}, n_ctx = {n_cxt}, k_kv_req = {n_kv_req}");

        // make sure the KV cache is big enough to hold all the prompt and generated tokens
        if n_kv_req > n_cxt {
            panic!(
            "n_kv_req > n_ctx, the required kv cache size is not big enough
either reduce n_len or increase n_ctx"
        )
        }

        if tokens_list.len() >= usize::try_from(args.max_token).unwrap() {
            panic!("the prompt is too long, it has more tokens than n_len")
        }
        

        // create a llama_batch with size ctx_size
        
        // we use this object to submit token data for decoding
        let mut batch = LlamaBatch::new(n_cxt as usize, 1);

        let last_index: i32 = (tokens_list.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
            // llama_decode will output logits only for the last token of the prompt
            let is_last = i == last_index;
            batch.add(token, i, &[0], is_last).unwrap();
        }

        ctx.decode(&mut batch)
            .with_context(|| "llama_decode() failed").unwrap();

        // main loop

        let mut n_cur = batch.n_tokens();
        let mut n_decode = 0;

        // let t_main_start = ggml_time_us();

        // The `Decoder`
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        
        let mut llm_output = String::new();
        
        while n_cur <= args.max_token {
            // sample the next token
            let candidates = ctx.candidates();

            let candidates_p = LlamaTokenDataArray::from_iter(candidates, false);

            // sample the most likely token
            let new_token_id = ctx.sample_token_greedy(candidates_p);

            // is it an end of stream?
            if model.is_eog_token(new_token_id) {
                // eprintln!();
                break;
            }

            let output_bytes = model.token_to_bytes(new_token_id, Special::Tokenize).unwrap();
            // use `Decoder.decode_to_string()` to avoid the intermediate buffer
            let mut output_string = String::with_capacity(32);
            let _decode_result = decoder.decode_to_string(&output_bytes, &mut output_string, false);
            
            print!("{output_string}");
            llm_output.push_str(output_string.as_str());
            std::io::stdout().flush().unwrap();

            batch.clear();
            batch.add(new_token_id, n_cur, &[0], true).unwrap();

            n_cur += 1;

            ctx.decode(&mut batch).with_context(|| "failed to eval").unwrap();

            n_decode += 1;
        }
        
        let llm_output_formatted = format!("{}<|eot_id|>", llm_output);
        
        input_to_model.push_str(llm_output_formatted.as_str());

        // eprintln!("\n");
        // 
        // let t_main_end = ggml_time_us();
        // 
        // let duration = Duration::from_micros((t_main_end - t_main_start) as u64);
        // 
        // eprintln!(
        //     "decoded {} tokens in {:.2} s, speed {:.2} t/s\n",
        //     n_decode,
        //     duration.as_secs_f32(),
        //     n_decode as f32 / duration.as_secs_f32()
        // );
        // 
        // println!("{}", ctx.timings());
        
    }

}
