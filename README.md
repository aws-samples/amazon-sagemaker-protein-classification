# Fine-tuning and deploying ProtBert Model for Protein Classification using Amazon SageMaker
 ## Contents
  - [Motivation](#motivation)
  - [What is ProtBert?](#what-is-protbert)
  - [Notebook Overview](#notebook-overview)
  - [Dataset](#dataset)
  - [Amazon SageMaker](#-amazon-sagemaker)
  - [How to run the code in Amazon SageMaker Studio?](#-how-to-run-the-code-in-amazon-sagemaker-studio)
    - [To log in from the SageMaker console](#to-log-in-from-the-sagemaker-console)
    - [Open a Studio notebook](#open-a-studio-notebook)
      - [To clone the repo](#to-clone-the-repo)
    - [To open a notebook](#to-open-a-notebook)
  - [References](#-references)
  - [Security](#security)
  - [License](#license)

## Motivation
**Proteins** are the key fundamental macromolecules governing in biological bodies. The study of protein localization is important to comprehend the function of protein and has great importance for drug design and other applications. It also plays an important role in characterizing the cellular function of hypothetical and newly discovered proteins [1]. There are several research endeavours that aim to localize whole proteomes by using high-throughput approaches [2–4]. These large datasets provide important information about protein function, and more generally global cellular processes. However, they currently do not achieve 100% coverage of proteomes, and the methodology used can in some cases cause mislocalization of subsets of proteins [5,6]. Therefore, complementary methods are necessary to address these problems. In this notebook, we will leverage Natural Language Processing (NLP) techniques for protein sequence classification. The idea is to interpret protein sequences as sentences and their constituent – amino acids –
as single words [7]. More specifically we will fine tune Pytorch ProtBert model from Hugging Face library.

---
## What is ProtBert?
ProtBert is a pretrained model on protein sequences using a masked language modeling (MLM) objective. It is based on Bert model which is pretrained on a large corpus of protein sequences in a self-supervised fashion. This means it was pretrained on the raw protein sequences only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those protein sequences [8]. For more information about ProtBert, see [`ProtTrans: Towards Cracking the Language of Life’s Code Through Self-Supervised Deep Learning and High Performance Computing`](https://www.biorxiv.org/content/10.1101/2020.07.12.199554v2.full).

---
## Notebook Overview
This example notebook focuses on fine-tuning the Pytorch ProtBert model and deploying it using Amazon SageMaker, which is the most comprehensive and fully managed machine learning service. With SageMaker, data scientists and developers can quickly and easily build and train machine learning models, and then directly deploy them into a production-ready hosted environment. 
During the training, we will leverage SageMaker distributed data parallel (SDP) feature which extends SageMaker’s training capabilities on deep learning models with near-linear scaling efficiency, achieving fast time-to-train with minimal code changes. We will also be able to observe the progress of the training using Sagmeaker Debugger which can also profile machine learning models, making it much easier to identify and fix training issues caused by hardware resource usage.

_**Note**_: Please select the Kernel as ` Python 3 (Pytorch 1.6 Python 3.6 CPU Optimized)`.

---
## Dataset
We are going to use a opensource public dataset of protein sequences available [here](http://www.cbs.dtu.dk/services/DeepLoc-1.0/data.php). The dataset is a `fasta file` composed by header and protein sequence. The header is composed by the accession number from Uniprot, the annotated subcellular localization and possibly a description field indicating if the protein was part of the test set. The subcellular localization includes an additional label, where S indicates soluble, M membrane and U unknown[9].
Sample of the data is as follows :
```
>Q9SMX3 Mitochondrion-M test
MVKGPGLYTEIGKKARDLLYRDYQGDQKFSVTTYSSTGVAITTTGTNKGSLFLGDVATQVKNNNFTADVKVST
DSSLLTTLTFDEPAPGLKVIVQAKLPDHKSGKAEVQYFHDYAGISTSVGFTATPIVNFSGVVGTNGLSLGTDV
AYNTESGNFKHFNAGFNFTKDDLTASLILNDKGEKLNASYYQIVSPSTVVGAEISHNFTTKENAITVGTQHAL>
DPLTTVKARVNNAGVANALIQHEWRPKSFFTVSGEVDSKAIDKSAKVGIALALKP"
```
A sequence in FASTA format begins with a single-line description, followed by lines of sequence data. The definition line (defline) is distinguished from the sequence data by a greater-than (>) symbol at the beginning. The word following the ">" symbol is the identifier of the sequence, and the rest of the line is the description.

## Amazon SageMaker
----
Amazon SageMaker is the most comprehensive and full managed machine learning service. With SageMaker, data scientists and developers can quickly and easily build and train machine learning models, and then directly deploy them into a production-ready hosted environment. It provides an integrated Jupyter authoring notebook instance for easy access to your data sources for exploration and analysis, so you don't have to manage servers. It also provides common machine learning algorithms that are optimized to run efficiently against extremely large data in a distributed environment. With native support for bring-your-own-algorithms and frameworks, SageMaker offers flexible distributed training options that adjust to your specific workflows. Deploy a model into a secure and scalable environment by launching it with a few clicks from SageMaker Studio or the SageMaker console.  We use Amazon SageMaker Studio for running the code, for more details see the [AWS documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/studio.html).

## How to run the code in Amazon SageMaker Studio? 
----
If you haven't used Amazon SageMaker Studio before, please follow the steps mentioned in [`Onboard to Amazon SageMaker Studio`](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio-onboard.html).

### To log in from the SageMaker console

- Onboard to Amazon SageMaker Studio. If you've already onboarded, skip to the next step.
- Open the SageMaker console.
- Choose Amazon SageMaker Studio.
- The Amazon SageMaker Studio Control Panel opens.
- In the Amazon SageMaker Studio Control Panel, you'll see a list of user names.
- Next to your user name, choose Open Studio.

### Open a Studio notebook
SageMaker Studio can only open notebooks listed in the Studio file browser. In this example we will `Clone a Git Repository in SageMaker Studio`.

#### To clone the repo

- In the left sidebar, choose the File Browser icon ( <img src='https://docs.aws.amazon.com/sagemaker/latest/dg/images/icons/File_browser_squid.png'> ).
- Choose the root folder or the folder you want to clone the repo into.
- In the left sidebar, choose the Git icon ( <img src='https://docs.aws.amazon.com/sagemaker/latest/dg/images/icons/Git_squid.png'>  ).
- Choose Clone a Repository.
- Enter the URI for the repo https://github.com/aws-samples/amazon-sagemaker-visual-transformer.
- Choose CLONE.
- If the repo requires credentials, you are prompted to enter your username and password.
- Wait for the download to finish. After the repo has been cloned, the File Browser opens to display the cloned repo.
- Double click the repo to open it.
- Choose the Git icon to view the Git user interface which now tracks the examples repo.
- To track a different repo, open the repo in the file browser and then choose the Git icon.

### To open a notebook

- In the left sidebar, choose the File Browser icon ( <img src='https://docs.aws.amazon.com/sagemaker/latest/dg/images/icons/File_browser_squid.png'> ) to display the file browser.
- Browse to a notebook file and double-click it to open the notebook in a new tab.

## References
----
- [1] Refining Protein Subcellular Localization (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1289393/)
- [2] Kumar A, Agarwal S, Heyman JA, Matson S, Heidtman M, et al. Subcellular localization of the yeast proteome. Genes Dev. 2002;16:707–719. [PMC free article] [PubMed] [Google Scholar]
- [3] Huh WK, Falvo JV, Gerke LC, Carroll AS, Howson RW, et al. Global analysis of protein localization in budding yeast. Nature. 2003;425:686–691. [PubMed] [Google Scholar]
- [4] Wiemann S, Arlt D, Huber W, Wellenreuther R, Schleeger S, et al. From ORFeome to biology: A functional genomics pipeline. Genome Res. 2004;14:2136–2144. [PMC free article] [PubMed] [Google Scholar]
- [5] Davis TN. Protein localization in proteomics. Curr Opin Chem Biol. 2004;8:49–53. [PubMed] [Google Scholar]
- [6] Scott MS, Thomas DY, Hallett MT. Predicting subcellular localization via protein motif co-occurrence. Genome Res. 2004;14:1957–1966. [PMC free article] [PubMed] [Google Scholar]
- [7] ProtTrans: Towards Cracking the Language of Life's Code Through Self-Supervised Deep Learning and High Performance Computing (https://www.biorxiv.org/content/10.1101/2020.07.12.199554v2.full.pdf)
- [8] ProtBert Hugging Face (https://huggingface.co/Rostlab/prot_bert)
- [9] DeepLoc-1.0: Eukaryotic protein subcellular localization predictor (http://www.cbs.dtu.dk/services/DeepLoc-1.0/data.php)

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

