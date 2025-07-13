import os
import json
import logging
import torch

import faiss
import numpy as np
import torch.nn.functional as F
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz

from .backend import query
from .utils.response import *
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import traceback

logger = logging.getLogger("automind")


class KnowledgeRetriever(ABC):
    def __init__(self, task_desc: str):
        self.task_desc = task_desc

    @abstractmethod
    def step(self):
        pass


class PapersRetriever(KnowledgeRetriever):
    """
    Semantic paper retriever that connects machine learning tasks with relevant research papers.

    Key Features:
    - Encodes paper abstracts and task descriptions using transformer models
    - Maintains FAISS index for efficient similarity search
    - Supports dynamic updating of paper databases
    - Provides multi-criteria retrieval based on task requirements

    Args:
        model: Generative LLMs for analyzing papers
        task_desc: Target task description
        encoding_model: Path to pretrained sentence encoding model, small LMs like Bert
        embedding_dim: Dimension of sentence embeddings
        device: Hardware device for computation (cpu/gpu)

    Methods:
        prepare_paperDB(): Load or preprocess paper metadata (generating the descriptions of each paper)
        PaperIndex(): Build search index from paper database (construct (paper file name, paper embedding) pair)
        step(): Execute retrieval pipeline with query expansion
    """

    def __init__(
        self,
        model,
        task_desc,
        paper_analysis_file="automind/mlkg/paper_analysis.json",
        encoding_model="automind/backend/all-MiniLM-L6-v2",
        embedding_dim=384,
        device="cpu",
    ):
        super().__init__(task_desc)

        self.model = model
        self.paper_analysis_file = paper_analysis_file
        self.tokenizer = AutoTokenizer.from_pretrained(encoding_model)
        self.encoding_model = AutoModel.from_pretrained(encoding_model)
        self.device = torch.device(device)
        self.encoding_model.to(self.device)
        self.embedding = faiss.IndexFlatL2(embedding_dim)
        self.index = []

    def encode_sentence(self, sentence):
        encoded_input = self.tokenizer(
            sentence, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            model_output = self.encoding_model(**encoded_input)
        token_embeddings = model_output[0]
        input_mask_expanded = (
            encoded_input["attention_mask"]
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )
        embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
        normalized_embedding = F.normalize(embedding, p=2, dim=1)
        return normalized_embedding

    def encoding(self, sentence_list):
        for sentence in sentence_list:
            cur_emb = self.encode_sentence(sentence)
            self.embedding.add(cur_emb)

    def search(self, query, topk=5):
        query_emb = self.encode_sentence(query)
        distances, indices = self.embedding.search(query_emb, topk)
        return distances, indices

    def get_new_paper_attribute(self, paper):
        keywords = {
            "Task": "The mechine learning task they want to solve (have corresponding datasets to evaluate the task preformance). e.g., classification, regression, segmentation, node/link/graph-level cls task",
            "Data_type": "The data they can handled, e.g., the node/edge information on graph-structured data (textual node attributes homogeneous graphs), image-textual multi-modal pictures, SMILES/fingerprint",
            "Data_domain": " The domain that data comes from, e.g., economic, academic, social network",
            "ML_phase": " The location in ML pipeline, e.g., detailed steps in data processing/generation, model architecture design, training framework",
        }

        prompt = (
            "You are an expert in machine learning. "
            "You need to give a brief of the paper on the given aspects:{}\n"
            "Notes: "
            "- Summary MUST clear and precise, using terminology consistent with ML domain and academic papers."
            "- Summary MUST strictly describe the tag's dimension. Do NOT generate irrelevant terms (e.g., describe the model architectures when meeting data type.)."
            "- At most two sentence."
            "Here is the paper information (abstract/introduction/conclusion): {}"
        )

        results = {}
        for key in keywords.keys():
            response = query(
                system_message=prompt.format(key + keywords[key], paper),
                user_message=None,
                model=self.model,
                temperature=0.5,
            )
            # response = extract_json(response)
            results[key] = response
        return results

    def prepare_paperDB(self, paper_dir="automind/mlkg/extraction"):
        try:
            with open(self.paper_analysis_file, "r") as f:
                papers = json.load(f)
                print(f"loaded paper from json: {self.paper_analysis_file}")
        except:
            papers = {}

        for root, subdirs, files in os.walk(paper_dir):
            for file in files:
                try:
                    paper_file = os.path.join(root, file)
                    with open(paper_file, "r") as f:
                        json_obj = json.load(f)
                    papers[file] = {
                        "title": json_obj["meta_info"]["title"],
                        "abs": json_obj["meta_info"]["abstract"],
                    }
                    # print(papers[file])
                    analysis = self.get_new_paper_attribute(
                        paper=json_obj["latex_extraction"]
                    )
                    papers[file].update(analysis)
                    print("-" * 30, file)
                    print(analysis)

                    # save after each paper extracted
                    with open(self.paper_analysis_file, "w") as f:
                        json.dump(papers, f, ensure_ascii=False, indent=4)
                except:
                    continue

    def paperIndex(self):
        """
        construct (paper file name, paper embedding) pair
        paperDB_file: the paperDB json file.
        """
        with open(self.paper_analysis_file, "r") as f:
            papers = json.load(f)

        # obtain paper index-embedding pair
        file_descriptions = []
        for file in papers.keys():
            self.index.append(file)
            desp = "\n".join([papers[file][key] for key in papers[file].keys()])
            file_descriptions.append(desp)
        self.encoding(file_descriptions)

    def generate_query(self):
        keywords = {
            "Task": "The mechine learning task they want to solve (have corresponding datasets to evaluate the task preformance). e.g., classification, regression, segmentation, node/link/graph-level cls task",
            "Data_type": "The data they can handled, e.g., the node/edge information on graph-structured data (textual node attributes homogeneous graphs), image-textual multi-modal pictures, SMILES/fingerprint",
            "Data_domain": " The domain that data comes from, e.g., economic, academic, social network",
            "ML_phase": " The location in ML pipeline, e.g., detailed steps in data processing/generation, model architecture design, training framework",
        }
        prompt = """
        I want to retrieve some papers for the given task from different perspectives.
        You are an expert in machine learning. You need to describe this task from the given prespective, and provide several keywords to describe this task.
        
        The task description: {} \n\n
        The given perspective: {}
        
        The response shoule be a sentence with no more than 50 words.
        """
        retreive_query = []
        for key in keywords.keys():
            response = query(
                system_message=prompt.format(self.task_desc, key + keywords[key]),
                user_message=None,
                model=self.model,
                temperature=0.5,
            )
            print("===" * 20, key)
            print(response)
            retreive_query.append(response)
        return retreive_query

    def step(self, retrieve_topk=4):
        """
        Retrieve topk paper based on query list, and then merge results.
        """
        self.paperIndex()
        merged_score = np.zeros(len(self.index))
        searched_results = []
        retrieve_query = self.generate_query()

        for q in retrieve_query:
            dis, indices = self.search(q, retrieve_topk)
            # re-score
            for idx, distance in zip(indices[0], dis[0]):
                searched_results.append((idx, 1 / (1 + distance)))

        # merge all searched papers in differet query
        all_searched_doc = {}
        for _, (idx, scores) in enumerate(searched_results):
            merged_score[idx] += scores
        final_selected = np.argsort(merged_score)[-retrieve_topk:][::-1]

        selected_plist = [self.index[i] for i in final_selected]
        selected_papers = {}
        for key in selected_plist:
            with open("automind/mlkg/extraction/" + key, "r") as f:
                # TODO: only abs/introduction/method utilized in plan generation
                content = json.load(f)["latex_extraction"]
                selected_papers[key] = content
        return selected_papers


class TricksRetriever(KnowledgeRetriever):
    def __init__(
        self,
        task_desc: str,
        model: str,
        embedding_model: str = "automind/backend/all-MiniLM-L6-v2",
        trick_corpus: str = "automind/utils/tagged_competition_solutions.jsonl",
        load_corpus=True,
        task_tag_corpus: str = "automind/utils/taggs_complete.json",
    ):
        super().__init__(task_desc)
        self.task_desc = task_desc.split("COMPETITION INSTRUCTIONS", 1)[-1].strip()
        logger.info(
            f"Loading tricks retriever with model: {model}, embedding model: {embedding_model}, corpus: {trick_corpus}"
        )
        self.model = model
        self.explicit_description = (
            task_desc.split("## Description", 1)[1].split("## Evaluation", 1)[0].strip()
            if "## Description" in task_desc and "## Evaluation" in task_desc
            else (
                task_desc.split("## Description", 1)[1].strip()
                if "## Description" in task_desc
                else (
                    task_desc.split("## Evaluation", 1)[0].strip()
                    if "## Evaluation" in task_desc
                    else task_desc.strip()
                )
            )
        )
        with open(task_tag_corpus, "r", encoding="utf-8") as f:
            self.task_tags = json.load(f)
        self.task_tags_description = {
            key: value["description"] for key, value in self.task_tags.items()
        }
        if load_corpus:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.embedding_model = SentenceTransformer(embedding_model).to(self.device)
            self.corpus = self._load_corpus(trick_corpus)
            self.embedded_corpus = self._embed_corpus(trick_corpus)

    def _load_corpus(self, corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus = []
            for line in f:
                obj = json.loads(line)
                corpus.append(obj)
        return corpus

    def _embed_corpus(self, corpus_path):
        # load competition
        competition_list = []
        for line in open(corpus_path, "r", encoding="utf-8"):
            obj = json.loads(line)
            competition_list.append(obj)

        # corpus initialization
        embedded_corpus = {}
        embedded_corpus["task"] = []
        hard_tag_list = list(
            set([key for d in competition_list for key in d.get("tags", {}).keys()])
        )
        embedded_corpus["label"] = {tag: [] for tag in hard_tag_list}

        for competition in competition_list:
            # task embedding
            if "introduction" in competition:
                text = (
                    competition["introduction"].split("\n== Evaluation ==")[0].strip()
                )
                embedding = self._get_embedding(text)
            else:
                print("No introduction found in the competition.")
                embedding = torch.zeros(
                    self.embedding_model.get_sentence_embedding_dimension()
                ).to(self.device)
            embedded_corpus["task"].append(embedding)

            # label embedding
            for hard_tag in embedded_corpus["label"].keys():
                if hard_tag not in competition["tags"].keys():
                    embedding = torch.zeros(
                        self.embedding_model.get_sentence_embedding_dimension()
                    ).to(self.device)
                else:
                    embedding = self._get_embedding(competition["tags"][hard_tag])
                embedded_corpus["label"][hard_tag].append(embedding)

        return embedded_corpus

    def _get_similarity(self, text1, text2):
        embedding1 = self._get_embedding(text1)
        embedding2 = self._get_embedding(text2)
        return self.embedding_model.similarity(embedding1, embedding2)

    def _get_embedding(self, text):
        return self.embedding_model.encode(text, convert_to_tensor=True).to(self.device)

    def _get_candidates_similarity(
        self, query: str, embedded_corpus: list, n: int = None
    ):
        query_embedding = self._get_embedding(query)
        # 确保embedded_corpus是张量
        if not torch.is_tensor(embedded_corpus[0]):
            embedded_corpus = torch.stack(
                [torch.tensor(e).to(self.device) for e in embedded_corpus]
            )
        else:
            embedded_corpus = torch.stack(embedded_corpus)

        embedding_similarity_matrix = torch.cosine_similarity(
            query_embedding.unsqueeze(0), embedded_corpus, dim=1
        )
        if n is None or n > len(embedded_corpus) or n <= 0:
            n = len(embedded_corpus)
        similar_indices = torch.argsort(embedding_similarity_matrix, descending=True)[
            :n
        ]
        similar_candidates = [
            (i.item(), embedding_similarity_matrix[i.item()].item())
            for i in similar_indices
        ]
        return similar_candidates, embedding_similarity_matrix

    def _get_tags_similarity(self, tags1, embedded_tags2):
        embedded_tags1 = self._get_embedding(tags1)

        # null vector check
        if (
            torch.all(embedded_tags1 == 0)
            or torch.all(embedded_tags2 == 0)
            or embedded_tags1.numel() == 0
            or embedded_tags2.numel() == 0
        ):
            return 0

        # center similarity
        query_center = torch.mean(embedded_tags1, dim=0)  # [dim]
        task_center = torch.mean(embedded_tags2, dim=0)  # [dim]
        center_sim = torch.cosine_similarity(
            query_center.unsqueeze(0), task_center.unsqueeze(0), dim=1
        )[0].item()

        # average similarity
        similarity_matrix = torch.cosine_similarity(
            embedded_tags1.unsqueeze(1), embedded_tags2.unsqueeze(0), dim=2
        )  # [num_tags1, num_tags2]
        max_similarities = torch.max(similarity_matrix, dim=1)[
            0
        ]  # 每个tag1对tag2的最大相似度
        best_match_sim = torch.mean(max_similarities).item()  # 平均最大相似度

        return 0.8 * center_sim + 0.2 * best_match_sim

    def _check_same_task(self, task1: str, task2: str):
        prompt = f"""
            # Task
            Here are two descriptions of machine learning competitions:
            Description 1: 
            {task1}
            Description 2:
            {task2}
            
            # Response Format
            Now please determine if these two competitions are exactly the same competition. Return "TRUE" if they are the same, return "FALSE" if they are different.
        """
        response = query(
            system_message=prompt,
            user_message=None,
            model=self.model,
            temperature=0.5,
        )
        if clean_string("TRUE") in clean_string(response):
            return True
        if clean_string("FALSE") in clean_string(response):
            return False
        return None

    def _get_relevant_tasks_tags(
        self, task_description: str, tags: dict, top_solutions: int = 5
    ):
        # description similarity
        logger.info(f"Fetching task description...")
        embedding_matches, desp_score = self._get_candidates_similarity(
            task_description, self.embedded_corpus["task"]
        )

        # check same task
        logger.info(f"Checking same task...")
        omit_index = (
            embedding_matches[0][0]
            if self._check_same_task(
                task_description,
                self.corpus[embedding_matches[0][0]]["introduction"]
                .split("\n== Evaluation ==")[0]
                .strip(),
            )
            else -1
        )

        # tags similarity
        logger.info(f"Calculating tag similarity...")
        tags_score = torch.zeros(len(self.corpus)).to(self.device)
        for hard_tag, soft_tag in tags.items():
            if hard_tag not in self.embedded_corpus["label"]:
                logger.info(f"Invalid tag {hard_tag}. Omitted.")
                continue
            for idx, embedded_tags in enumerate(
                self.embedded_corpus["label"][hard_tag]
            ):
                similarity = self._get_tags_similarity(soft_tag, embedded_tags)
                tags_score[idx] += torch.tensor(similarity, device=self.device)

        # Normalize and combine scores
        desp_score = (
            desp_score / desp_score.max() if desp_score.max() > 0 else desp_score
        )
        tags_score = (
            tags_score / tags_score.max() if tags_score.max() > 0 else tags_score
        )

        final_scores = 0.2 * desp_score + 0.8 * tags_score

        # Using torch.topk instead of numpy sorting
        values, indices = torch.topk(final_scores, len(final_scores))

        # Get top matching tasks
        total_solutions = {}
        logger.info("\nTop 5 Similar:")
        for score, idx in zip(values[:5], indices[:5]):
            logger.info(
                f"{self.corpus[idx]['title']} | Score: {score:.4f} | Tags: {', '.join(self.corpus[idx]['tags'])}"
            )

        results = []
        total_solutions = {}

        for idx in indices:
            idx = idx.item()
            if idx == omit_index:
                continue

            current_solutions = {}
            for solution in self.corpus[idx]["solutions"].items():
                current_solutions[solution[0]] = solution[1]
                total_solutions[solution[0]] = solution[1]
                if len(total_solutions) >= top_solutions:
                    break

            if current_solutions:
                results.append(
                    {"title": self.corpus[idx]["title"], "solutions": current_solutions}
                )

            if len(total_solutions) >= top_solutions:
                results[-1]["solutions"] = dict(
                    list(results[-1]["solutions"].items())[
                        : top_solutions
                        - len(total_solutions)
                        + len(results[-1]["solutions"])
                    ]
                )
                break

        return results

    @property
    def _prompt_extract_response_format(self):
        return {
            "Response format": (
                "First, thoroughly analyze the given information and engage in detailed reasoning.\n"
                "Then, select the most appropriate categories based on the following rules:\n"
                "- Choose the most relevant matching categories\n"
                '- Only use "General Machine Learning" if no other categories are appropriate\n'
                '- Avoid "General Machine Learning" if more specific categories exist\n\n'
                'Return your category selection as a JSON dictionary with key "category" and a List[str] value.\n'
                "At the end of your response, provide a JSON dictionary (wrapped in ```)."
            )
        }

    @property
    def _prompt_extract_soft_label_format(self):
        return {
            "Response format": (
                "First, thoroughly analyze the given information and engage in detailed reasoning.\n"
                "Then, select the most appropriate categories based on the following rules:\n"
                "- Choose the most relevant matching categories\n"
                "- If multiple categories are equally relevant, combine them with commas\n"
                "- If no existing categories match well, provide a custom descriptive tag\n\n"
                'Return your category selection as a JSON dictionary with key "category" and a string value.\n'
                "At the end of your response, provide a JSON dictionary (wrapped in ```)."
            )
        }

    @property
    def _prompt_distill_response_format(self):
        return {
            "Response format": """
                Your response should contain your analysis of the useful tricks in the solutions. 
                
                The tricks should be structured as follows:
                [Applicable Stage]: [Specific content and concrete practices of the transferable knowledge. ]
                ...
                [Applicable Stage]: [Specific content and concrete practices of the transferable knowledge. ]
                Please analyze the useful tricks for improving results, ignoring the ineffective ones.
                
                Note that:
                1. Ensure that each trick represents independent and precise information. 
                2. Preserve as much original information as possible, while omitting numerical values. 
                3. Remove those tricks that require external resources (such as datasets downloaded from the internet, paid APIs or public leaderboard, etc.).
                4. It is recommended to extract 8 to 12 tricks. Each trick item should be detailed and maintain as much of the original information as possible. 
                5. As long as the solution content indicates that the trick enhances the experimental results, the more tricks you can extract, the better.
                
                At the end of your response, provide the useful tricks formatted as a json dictionary (wrapped in ```). The key denotes the applicable stage, and the value is a string detailing the implementation specifics of that Trick.
            """
        }

    @property
    def _prompt_restruct_response_format(self):
        return {
            "Response format": """
                Your response should include a detailed analysis of the current task, as well as which tricks must be used together to be effectively applied in the target task.

                The restructed tricks should be structured as dictionary list as follows:
                [
                    {{
                        "applyable stage": "trick content",    
                        "applyable stage": "trick content",    
                    }},
                    ...
                    {{
                         "applyable stage": "trick content",    
                    }}
                ]
                Each group of tricks is presented as a dictionary. 
                
                **IMPORTANT**:
                1. Ensure that NO given tricks are duplicated after restructed. Each trick is exclusive to a single trick group and will not appear in any other group.
                2. Remove tricks requiring private external resources (like private datasets or paid APIs), while allowing use of public Hugging Face models and datasets.
                3. Remove those tricks that are completely unsuitable or inapplicable for the target task.
                4. Maximize the retention of the details of the provided tricks. Only make necessary modifications to adapt them for the target task.
                
                At the end of your response, provide the restructed trick groups formatted as a json dictionary (wrapped in ```), with the key "restructed_tricks", and the value being the trick group list formatted as described above.
            """
        }

    @property
    def _prompt_restruct_response_format_new(self):
        return {
            "Response format": """
                Your response should include a detailed analysis of the current task, as well as analysis and evaluation of each trick listed above.

                The restructed tricks should be structured as a json dictionary as follows:
                {{
                    "restructed_tricks": {{
                        "applyable stage": "trick content",    
                        "applyable stage": "trick content",    
                    }}
                }}
                Each group of tricks is presented as a dictionary. 
                
                **IMPORTANT**:
                1. Remove invalid tricks requiring private external resources (like private datasets or paid APIs), while allowing use of public Hugging Face models and datasets.
                2. Remove those tricks that are completely unsuitable or inapplicable for the target task.
                3. Prohibit ANY form of deletion or summarization of the valid trick content. Every effective detail of the original trick MUST be fully preserved. Adjust incompatible details or refine overly broad aspects for target task adaptation when necessary.
                4. Permit direct merging of overly similar tricks or highly related tricks that must be used together within the workflow. Ensure maximum retention of all original information WITHOUT summarization or refinement.

                Now analyze each Trick individually, maximizing retention of effective entries and their content. 
                At the end of your response, provide the restructed tricks formatted as a json dictionary (wrapped in ```), with the key "restructed_tricks", and the value being the filtered tricks formatted as described above.
            """
        }

    def extract_prompt_integration(self):
        prompt = {}
        prompt["Introduction"] = (
            "Given the following machine learning competition, perform analysis, reasoning, and synthesis for this competition.\n"
            "Here is the competition description:\n"
            f"{self.task_desc}\n"
            "Based on the competition description above, let's outline its category. Here is the list of categories and some descriptions:\n"
            f"{self.task_tags_description}\n"
            "Now choose the task category from the list below that best fits the competition description.\n"
            f"{self.task_tags.keys()}\n"
        )
        prompt["Instructions"] = {}
        prompt["Instructions"] |= self._prompt_extract_response_format
        return prompt

    def extract_soft_label_prompt_integration(self, hard_tag, soft_tag):
        prompt = {}
        prompt["Introduction"] = (
            "Given the following machine learning competition, perform analysis, reasoning, and synthesis for this competition.\n"
            "Here is the competition description:\n"
            f"{self.task_desc}\n"
            f"The competition has already been categorized as: {hard_tag}\n"
            "Next are the subcategories of technologies within this field: \n"
            f"{soft_tag}\n"
            "Now choose the suitable technical category. \n"
            "1. Choose as few categories as possible while ensuring quality. \n"
            "2. Only provide a custom category using machine learning terminology if no suitable option exists. \n"
        )
        prompt["Instructions"] = {}
        prompt["Instructions"] |= self._prompt_extract_soft_label_format
        return prompt

    def distill_prompt_integration(self, solutions):
        prompt = {}
        prompt["Introduction"] = (
            "You are an expert in machine learning. "
            "Based on given tasks and some top solutions, filter out useful knowledge that can help improve experimental results.\n"
            "Preserve as many implementation details as possible, so that this knowledge can be applied to similar tasks in the future.\n\n"
            "Here are the tasks and their solutions:\n"
            f"{solutions}\n"
        )
        prompt["Instructions"] = {}
        prompt["Instructions"] |= self._prompt_distill_response_format
        return prompt

    def restruct_tricks_prompt_integration(self, tricks: dict):
        prompt = {}
        prompt["Introduction"] = (
            "You are an expert in machine learning. "
            "Your task is to enrich and reorganize some tricks based on the given task. \n"
            "Here is the target task:\n"
            f"{self.task_desc}\n"
        )
        prompt["Tricks"] = (
            "Here are some useful tricks distilled from other tasks:\n"
            f"{tricks}\n"
            "Now group the interdependent tricks that must be used together."
        )
        prompt["Instructions"] = {}
        prompt["Instructions"] |= self._prompt_restruct_response_format
        return prompt

    def restruct_tricks_prompt_integration_new(self, tricks: dict):
        prompt = {}
        prompt["Introduction"] = (
            "You are an expert in machine learning. "
            "Your task is to filter and enrich some tricks based on the given task. \n"
            "Here is the target task:\n"
            f"{self.task_desc}\n"
        )
        prompt["Tricks"] = (
            "Here are some useful tricks distilled from other tasks:\n"
            f"{tricks}\n"
            "Now filter and reconstruct the useful tricks, which are applicable to the current task and can significantly enhance the results."
        )
        prompt["Instructions"] = {}
        prompt["Instructions"] |= self._prompt_restruct_response_format_new
        return prompt

    def step(self):
        hard_prompt = self.extract_prompt_integration()
        response = query(
            system_message=hard_prompt,
            user_message=None,
            model=self.model,
            temperature=0.5,
        )
        tags = {}
        final_response = extract_json_dict(response)
        task_categories = final_response["category"]
        logger.info(f"Task categories: {task_categories}")
        if isinstance(task_categories, str):
            task_categories = [task_categories]

        for category in task_categories:
            if category not in self.task_tags.keys():
                logger.error(f"Invalid task category: {category}.")
                continue
            soft_tags = self.task_tags[category]["category"]
            soft_prompt = self.extract_soft_label_prompt_integration(
                category, soft_tags
            )
            response = query(
                system_message=soft_prompt,
                user_message=None,
                model=self.model,
                temperature=0.5,
            )
            soft_response = extract_json_dict(response)["category"]
            soft_tags = [item.strip() for item in soft_response.split(",")]
            soft_tags_filtered = (
                [
                    item
                    for item in soft_tags
                    if item in self.task_tags[category]["category"]
                ]
                if any(
                    item in self.task_tags[category]["category"] for item in soft_tags
                )
                else soft_tags
            )
            tags[category] = soft_tags_filtered

            logger.info(f"Soft categories: {task_categories}")

        # retrieve tasks
        try:
            relevant_competitions = self._get_relevant_tasks_tags(
                task_description=self.explicit_description, tags=tags
            )
            if relevant_competitions is None or len(relevant_competitions) == 0:
                logger.info(f"No relevant competitions retrieved.")
                return None
            logger.info(f"Relevant competitions: {relevant_competitions}")
        except Exception as e:
            logger.error(f"Error retrieving relevant competitions: {e}")
            logging.error(f"Full traceback: {traceback.format_exc()}")

        # distill tricks
        distilled_tricks = {}
        prompt = self.distill_prompt_integration(json.dumps(relevant_competitions))
        response = query(
            system_message=prompt,
            user_message=None,
            model=self.model,
            temperature=0.5,
        )
        distilled_tricks = extract_json_dict(response)
        if distilled_tricks is None or len(distilled_tricks) == 0:
            logger.info(f"No tricks extracted.")
            return None
        logger.info(f"Useful tricks: {distilled_tricks}")

        # restruct tricks
        # prompt = self.restruct_tricks_prompt_integration(distilled_tricks)
        prompt = self.restruct_tricks_prompt_integration_new(distilled_tricks)
        response = query(
            system_message=prompt,
            user_message=None,
            model=self.model,
            temperature=0.5,
        )
        restructed_tricks = extract_json_dict(response)
        try:
            trick_groups = restructed_tricks["restructed_tricks"]
            logger.info(f"Restructed tricks: {json.dumps(trick_groups, indent=2)}")
            if not isinstance(trick_groups, list):
                trick_groups = [{key: value} for key, value in trick_groups.items()]
        except Exception as e:
            logger.error(f"Error restructing competitions: {e}")
            logging.error(f"Full traceback: {traceback.format_exc()}")
            trick_groups = [{key: value} for key, value in distilled_tricks.items()]

        initial_tricks = [{"content": d, "invalid": False} for d in trick_groups]

        return initial_tricks
