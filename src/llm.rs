use std::ffi::CString;
use std::io::Write;
use std::num::NonZeroU32;
use std::pin::pin;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use std::sync::Arc;

use super::args_handler::{Args, Mode};

pub struct LLM {
    pub model: Arc<LlamaModel>,
    pub backend: LlamaBackend,
    pub ctx_params: LlamaContextParams,
    mode: Mode,
    history: String,
    max_token: i32,
}

impl LLM {

    pub fn new(args: Args) -> Self{
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
            let k = CString::new(k.as_bytes()).expect(format!("invalid key: {k}").as_str());
            model_params.as_mut().append_kv_override(k.as_c_str(), *v);
        }

        let model_path = args.model.clone()
            .get_or_load()
            .expect("failed to get model from args");

        

        // Load the model and wrap it in an Arc for shared ownership
        let model = Arc::new(LlamaModel::load_from_file(&backend, model_path, &model_params)
                .expect("failed to load model"
        ));

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
        
        Self {
            model,
            backend,
            ctx_params,
            mode: args.mode,
            history: String::new(),
            max_token: args.max_token,
        }

        
    }
    
    pub fn generate_chat(&mut self, ctx: &mut LlamaContext, prompt: &str) -> String {

        


        let input_formatted = format!("<|start_header_id|>user<|end_header_id|>{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>", prompt);

        self.history.push_str(input_formatted.as_str());


        // tokenize the prompt
        // TODO: do not tokenize the prompt every time, add to the end of the token list

        let tokens_list = ctx.model
            .str_to_token(&self.history, AddBos::Never)
            .expect(format!("failed to tokenize {}", self.history).as_str());

        let n_cxt = ctx.n_ctx() as i32;
        let max_token = self.max_token;
        // let max_token = args.max_token;

        // eprintln!("n_len = {max_token}, n_ctx = {n_cxt}, k_kv_req = {n_kv_req}");

        // make sure the KV cache is big enough to hold all the prompt and generated tokens
        if max_token > n_cxt {
            panic!(
                "n_kv_req > n_ctx, the required kv cache size is not big enough
either reduce n_len or increase n_ctx"
            )
        }

        if tokens_list.len() >= max_token as usize{
            panic!("the prompt is too long, it has more tokens than max_token")
        }


        // create a llama_batch with size ctx_size

        // we use this object to submit token data for decoding
        let mut batch = LlamaBatch::new(n_cxt as usize, 1);

        let last_index = tokens_list.len() - 1;
        for (i, token) in tokens_list.into_iter().enumerate() {
            // llama_decode will output logits only for the last token of the prompt
            let is_last = i == last_index;
            batch.add(token, i as i32, &[0], is_last).unwrap();
        }

        ctx.decode(&mut batch)
            .expect("llama_decode() failed");

        // main loop

        let mut n_cur = batch.n_tokens();
        let mut _n_decode = 0;

        // let t_main_start = ggml_time_us();

        // The `Decoder`
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        let mut llm_output = String::new();

        while n_cur <= self.max_token {
            // sample the next token
            let candidates = ctx.candidates();

            let candidates_p = LlamaTokenDataArray::from_iter(candidates, false);

            // sample the most likely token
            let new_token_id = ctx.sample_token_greedy(candidates_p);

            // is it an end of stream?
            if ctx.model.is_eog_token(new_token_id) {
                // eprintln!();
                break;
            }

            let output_bytes = ctx.model.token_to_bytes(new_token_id, Special::Tokenize).unwrap();
            // use `Decoder.decode_to_string()` to avoid the intermediate buffer
            let mut output_string = String::with_capacity(32);
            let _decode_result = decoder.decode_to_string(&output_bytes, &mut output_string, false);

            print!("{output_string}");
            llm_output.push_str(output_string.as_str());
            std::io::stdout().flush().unwrap();

            batch.clear();
            batch.add(new_token_id, n_cur, &[0], true).unwrap();

            n_cur += 1;

            ctx.decode(&mut batch).expect("failed to eval");

            _n_decode += 1;
        }

        let llm_output_formatted = format!("{}<|eot_id|>", llm_output);

        self.history.push_str(llm_output_formatted.as_str());
        llm_output

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