#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

mod args_handler;
mod llm;

use args_handler::Args;
use llm::LLM;
use clap::{Parser};
use std::io::Write;


fn main() {

    let args: Args = Args::parse();

    let mut rllm: LLM = LLM::new(args);
    
    // TODO: how to move this inside the LLM struct?
    let binding = rllm.model.clone();
    let mut ctx = binding
        .new_context(&rllm.backend, rllm.ctx_params.clone())
        .expect("failed to create context");


    // main loop
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
    
        rllm.generate_chat(&mut ctx, &input);
    }

}
