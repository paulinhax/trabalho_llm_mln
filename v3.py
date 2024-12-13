import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from bert_score import BERTScorer
from rouge_score import rouge_scorer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Função para carregar os datasets de QA (pergunta e resposta)
def load_qa_datasets():
    """
    Carrega os datasets do formato parquet e salva em CSV
    Datasets incluídos: Titanic, Holiday, Disneyland e Central
    Cria três versões: completa (all), amostra (sample) e perguntas (qa)
    """
    # Cria diretórios se não existirem
    os.makedirs("training_all", exist_ok=True)
    os.makedirs("training_sample", exist_ok=True)
    os.makedirs("questions", exist_ok=True)
    
    dataset_ids = ['002_Titanic', '016_Holiday', '061_Disneyland', '009_Central']
    datasets = {}
    
    for id_dataset in dataset_ids:
        # Carrega e salva versão completa (all)
        df_all = pd.read_parquet(f"hf://datasets/cardiffnlp/databench/data/{id_dataset}/all.parquet")
        df_all.to_csv(f"training_all/{id_dataset}_all.csv", sep=";")
        
        # Carrega e salva versão de amostra (sample)
        df_sample = pd.read_parquet(f"hf://datasets/cardiffnlp/databench/data/{id_dataset}/sample.parquet")
        df_sample.to_csv(f"training_sample/{id_dataset}_sample.csv", sep=";")
        
        # Carrega e salva versão QA
        df_qa = pd.read_parquet(f"hf://datasets/cardiffnlp/databench/data/{id_dataset}/qa.parquet")
        df_qa.to_csv(f"questions/{id_dataset}_qa.csv", sep=";")
        
        # Armazena os DataFrames
        datasets[id_dataset] = {
            'all': df_all,
            'sample': df_sample,
            'qa': df_qa
        }
    
    return datasets

def load_saved_datasets():
    """
    Carrega os datasets já salvos em CSV
    Retorna: dicionário com DataFrames para cada dataset e versão
    """
    dataset_ids = ['002_Titanic', '016_Holiday', '061_Disneyland', '009_Central']
    datasets = {}
    
    for id_dataset in dataset_ids:
        try:
            # Carrega dados dos arquivos CSV
            df_all = pd.read_csv(f"training_all/{id_dataset}_all.csv", sep=";")
            df_sample = pd.read_csv(f"training_sample/{id_dataset}_sample.csv", sep=";")
            df_qa = pd.read_csv(f"questions/{id_dataset}_qa.csv", sep=";")
            
            datasets[id_dataset] = {
                'all': df_all,
                'sample': df_sample,
                'qa': df_qa
            }
        except FileNotFoundError:
            print(f"Arquivos para {id_dataset} não encontrados. Execute load_qa_datasets() primeiro.")
    
    return datasets

class SimpleGPTQA:
    """
    Classe principal para o modelo de QA baseado em GPT
    Implementa um sistema simples de pergunta e resposta usando GPT-2
    """
    def __init__(self, model_name='gpt2'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Configura o padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        
        # Adiciona tokens especiais para formatar entrada de QA
        special_tokens = {
            'additional_special_tokens': ['<|question|>', '<|answer|>', '<|context|>'],
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Define limites de tokens
        self.max_length = 1024  # Limite máximo do GPT-2
        self.max_context = 512  # Reserva metade para contexto
        self.max_new_tokens = 100  # Limite para geração de resposta

    def chunk_context(self, context, max_length=512):
        """
        Divide o contexto em chunks menores
        """
        # Tokeniza o contexto completo
        tokens = self.tokenizer.encode(context, add_special_tokens=False)
        
        # Divide em chunks
        chunks = []
        for i in range(0, len(tokens), max_length):
            chunk = tokens[i:i + max_length]
            chunks.append(self.tokenizer.decode(chunk))
            
        return chunks

    def prepare_input(self, question, context_chunk):
        """
        Prepara a entrada para o modelo com um chunk específico do contexto
        """
        template = f"<|context|>{context_chunk}<|question|>{question}<|answer|>"
        
        # Tokeniza com padding e attention mask
        encoded = self.tokenizer(
            template,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding=True,
            pad_to_max_length=True
        )
        
        return encoded.to(self.device)

    def generate_answer(self, question, context):
        """
        Gera uma resposta para a pergunta usando chunks do contexto
        """
        # Divide o contexto em chunks menores
        context_chunks = self.chunk_context(context, self.max_context)
        
        # Lista para armazenar respostas de cada chunk
        all_outputs = []
        
        # Processa cada chunk
        for chunk in context_chunks:
            try:
                # Prepara input com o chunk atual
                inputs = self.prepare_input(question, chunk)
                
                # Gera resposta
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=True,
                    temperature=0.7
                )
                
                # Decodifica e extrai resposta
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer = generated_text.split('<|answer|>')[-1].strip()
                
                if answer:  # Se encontrou uma resposta não vazia
                    all_outputs.append(answer)
            
            except Exception as e:
                print(f"Erro ao processar chunk: {str(e)}")
                continue
        
        # Se não conseguiu gerar nenhuma resposta
        if not all_outputs:
            return "Não foi possível gerar uma resposta."
        
        # Retorna a resposta mais longa (geralmente a mais informativa)
        return max(all_outputs, key=len)

def convert_table_to_text(df, max_rows=50):
    """
    Converte DataFrame para texto, limitando o número de linhas para controlar o tamanho
    """
    text = ""
    
    # Limita o número de linhas processadas
    df_sample = df.head(max_rows)
    
    # Converte linhas para texto
    for _, row in df_sample.iterrows():
        for col in df_sample.columns:
            text += f"{col} is {row[col]}. "
    
    # Adiciona resumos das colunas (usando apenas a amostra)
    for col in df_sample.columns:
        text += f"The complete list of {col} contains {len(df[col].unique())} unique values. "
    
    return text

class Evaluator:
    """
    Classe para avaliar a qualidade das respostas geradas
    Implementa métricas ROUGE e BERTScore
    """
    def __init__(self):
        # Inicializa os scorers
        self.bert_scorer = BERTScorer(model_type='bert-base-uncased')
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def calc_rouge(self, predictions, ground_truth):
        """
        Calcula métricas ROUGE para as previsões
        Retorna scores médios e individuais para ROUGE-1, ROUGE-2 e ROUGE-L
        """
        scores = {
            'rouge1': {'p': [], 'r': [], 'f': []},
            'rouge2': {'p': [], 'r': [], 'f': []},
            'rougeL': {'p': [], 'r': [], 'f': []}
        }
        
        # Calcula scores para cada par de previsão/verdade
        for pred, true in zip(predictions, ground_truth):
            rouge_scores = self.rouge_scorer.score(str(true), str(pred))
            for metric in scores.keys():
                scores[metric]['p'].append(getattr(rouge_scores[metric], 'precision'))
                scores[metric]['r'].append(getattr(rouge_scores[metric], 'recall'))
                scores[metric]['f'].append(getattr(rouge_scores[metric], 'fmeasure'))
        
        # Calcula médias
        avg_scores = {
            metric: {
                key: np.mean(values) for key, values in metric_scores.items()
            } for metric, metric_scores in scores.items()
        }
        
        return avg_scores, scores

    def calc_bertscore(self, predictions, ground_truth):
        """
        Calcula BERTScore para as previsões
        Retorna precisão, recall e F1 médios e individuais
        """
        # Converte para string se necessário
        predictions = [str(p) for p in predictions]
        ground_truth = [str(t) for t in ground_truth]
        
        precision, recall, f1 = self.bert_scorer.score(predictions, ground_truth)
        
        return {
            'precision': precision.mean().item(),
            'recall': recall.mean().item(),
            'f1': f1.mean().item()
        }, {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist()
        }

    def plot_metrics(self, rouge_scores, bert_scores, dataset_name):
        """
        Gera visualizações para as métricas ROUGE e BERTScore
        Parâmetros:
            rouge_scores: scores ROUGE calculados
            bert_scores: scores BERT calculados
            dataset_name: nome do dataset para o título
        Retorna: figura matplotlib
        """
        # Cria subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plota scores ROUGE
        rouge_data = []
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            for score_type in ['p', 'r', 'f']:
                rouge_data.append({
                    'Metric': f'{metric}-{score_type}',
                    'Score': rouge_scores[0][metric][score_type]
                })
        
        rouge_df = pd.DataFrame(rouge_data)
        sns.barplot(data=rouge_df, x='Metric', y='Score', ax=ax1)
        ax1.set_title(f'ROUGE Scores - {dataset_name}')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plota scores BERT
        bert_data = []
        for metric, value in bert_scores[0].items():
            bert_data.append({
                'Metric': f'BERT-{metric}',
                'Score': value
            })
        
        bert_df = pd.DataFrame(bert_data)
        sns.barplot(data=bert_df, x='Metric', y='Score', ax=ax2)
        ax2.set_title(f'BERTScore Metrics - {dataset_name}')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig

def main():
    """
    Função principal que executa todo o pipeline de QA e avaliação
    Processa tanto os dados completos (all) quanto a amostra (sample)
    """
    #try:
        #print("Carregando datasets salvos...")
        #datasets = load_saved_datasets()
    #except FileNotFoundError:
    print("Baixando datasets do HuggingFace...")
    datasets = load_qa_datasets()
    
    qa_model = SimpleGPTQA()
    evaluator = Evaluator()
    
    # Processa primeiro a versão sample
    print("\nProcessando versão: SAMPLE")
    all_results_sample = {}
    
    for dataset_id, data_dict in datasets.items():
        print(f"\nProcessando {dataset_id} (sample)...")
        
        train_df = data_dict['sample']
        qa_df = data_dict['qa']
        
        # Converte dados com limite de linhas
        context = convert_table_to_text(train_df, max_rows=50)
        
        questions = qa_df['question'].tolist()
        correct_answers = qa_df['answer'].tolist()
        
        predictions = []
        for i, question in enumerate(questions):
            print(f"Processando pergunta {i+1}/{len(questions)}")
            try:
                answer = qa_model.generate_answer(question, context)
                predictions.append(answer)
            except Exception as e:
                print(f"Erro ao processar pergunta: {str(e)}")
                predictions.append("Erro na geração da resposta")
        
        rouge_scores = evaluator.calc_rouge(predictions, correct_answers)
        bert_scores = evaluator.calc_bertscore(predictions, correct_answers)
        
        all_results_sample[dataset_id] = {
            'predictions': predictions,
            'ground_truth': correct_answers,
            'rouge_scores': rouge_scores,
            'bert_scores': bert_scores
        }
        
        fig = evaluator.plot_metrics(rouge_scores, bert_scores, f"{dataset_id} (sample)")
        fig.savefig(f'{dataset_id}_sample_metrics.png')
        plt.close(fig)
        
        print(f"\nResultados para {dataset_id} (sample):")
        print("\nScores ROUGE:")
        for metric, scores in rouge_scores[0].items():
            print(f"{metric}: {scores}")
        print("\nScores BERT:")
        for metric, score in bert_scores[0].items():
            print(f"{metric}: {score:.4f}")
    
    # Processa a versão all
    print("\nProcessando versão: ALL")
    all_results_complete = {}
    
    for dataset_id, data_dict in datasets.items():
        print(f"\nProcessando {dataset_id} (all)...")
        
        train_df = data_dict['all']
        qa_df = data_dict['qa']
        
        # Converte dados com limite de linhas (maior para versão completa)
        context = convert_table_to_text(train_df, max_rows=100)
        
        questions = qa_df['question'].tolist()
        correct_answers = qa_df['answer'].tolist()
        
        predictions = []
        for i, question in enumerate(questions):
            print(f"Processando pergunta {i+1}/{len(questions)}")
            try:
                answer = qa_model.generate_answer(question, context)
                predictions.append(answer)
            except Exception as e:
                print(f"Erro ao processar pergunta: {str(e)}")
                predictions.append("Erro na geração da resposta")
        
        rouge_scores = evaluator.calc_rouge(predictions, correct_answers)
        bert_scores = evaluator.calc_bertscore(predictions, correct_answers)
        
        all_results_complete[dataset_id] = {
            'predictions': predictions,
            'ground_truth': correct_answers,
            'rouge_scores': rouge_scores,
            'bert_scores': bert_scores
        }
        
        fig = evaluator.plot_metrics(rouge_scores, bert_scores, f"{dataset_id} (all)")
        fig.savefig(f'{dataset_id}_all_metrics.png')
        plt.close(fig)
        
        print(f"\nResultados para {dataset_id} (all):")
        print("\nScores ROUGE:")
        for metric, scores in rouge_scores[0].items():
            print(f"{metric}: {scores}")
        print("\nScores BERT:")
        for metric, score in bert_scores[0].items():
            print(f"{metric}: {score:.4f}")

if __name__ == "__main__":
    main()