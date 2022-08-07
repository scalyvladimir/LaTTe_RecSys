# latte_recsys
Storage for implementations used in "Tensor-based Collaborative Filtering With Smooth Ratings Scale paper"(*place for arxiv link*).

ANONYMOUS AUTHOR(S)

**Problem statement:**

While there are many possible inconsistencies in users' perception, which is not considered in the latest RecSys papers, this work aims to solve the problem of different perception of the rating scale by users. Consider the following example. For some people giving a 5 star rating to an item is an extraordinary event, they do it rarely and only in exceptional cases. Other users are more generous and most of their ratings are 4 and 5 stars. This leads to a situation where the recommender system treats different ratings as different signals which may be misleading. For example, given two users where the first one rated three last films as 3, 4, 4 and another user rated exactly the same films as 4, 5, 5, we should understand that these are the same sets of preferences. To tackle this problem, this work imposes some notion of similarity. Our proposed model LaTTe(latent + attention) tested against baseline approaches together with one, based on original [paper](https://arxiv.org/abs/1802.05814).

This codebase is designed for tuning and training all the models, mentioned in "Tensor-based Collaborative Filtering With Smooth Ratings Scale paper". It is particularly based on [Polara framework](https://github.com/evfro/polara.git).

<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/anonymouspap/latte_recsys.git
   ```
   
2. Install all the demanded packages with:
   ```sh
   pip3 install -r requirements.txt
   ```
   
### Execution
   
1. To start tuning+training+testing process of ALL models run our code in the following way:
  
2. Pick dataset(*one of the following*) accoding to the hints in code: ["Movielens_1M", "Movielens_10M", "Video_Games", "CDs_and_Vinyl", "Electronics", "Video_Games_nf"], *eg.* Movilens_1M.

3. Run the script in the following fashion: 
  ```sh
   python3 main.py Movilens_1M > output.txt
   ```

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors

<p align="right">(<a href="#top">back to top</a>)</p>
