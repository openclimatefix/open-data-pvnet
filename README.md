# 🌞 Open Source Solar Forecasting Project – Volunteers Welcome! 🌞
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-3-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

[![ease of contribution: easy](https://img.shields.io/badge/ease%20of%20contribution:%20easy-32bd50)](https://github.com/openclimatefix/ocf-meta-repo?tab=readme-ov-file#how-easy-is-it-to-get-involved)

## Scope

We're building an open-source solar forecasting pipeline to integrate with the OCF's [PVNet](https://github.com/openclimatefix/pvnet) using publicly available data to predict solar generation at the national level, starting with the UK. Eventually, we aim to achieve **global coverage**! 🌍

Tasks include:
- Identifying gridded Numerical Weather Prediction (NWP) datasets.
- Downloading and transforming NWP data into the preferred Zarr format.
- Acquiring solar generation target data via APIs (e.g., PVlive's solar generation and capacity API).
- Creating pipelines for batching data and ML model experimentation.
- Deploying the model to the cloud.

We will begin in the UK to benchmark against OCF results and expand to other countries as the project progresses. 😄

---

## Installation

To use the `open-data-pvnet` library, you can install it via `pip`:

### Basic Installation

For most users:
```bash
pip install open-data-pvnet
```

---

## Volunteer Skills/Roles Needed

We are looking for volunteers with the following skills:
- **Data Engineers**: For data acquisition, curation,transformation, and storage.
- **Machine Learning Enthusiasts**: To experiment, train, and evaluate models.
- **Software Developers**: Especially those with Python expertise.
- **Cloud Computing Experts**: For deploying and scaling the model.
- **DevOps Specialists**: To streamline workflows and maintain infrastructure.

---

## Getting Started

Ready to dive in? Check out our [Getting Started Guide](getting_started.md) for an introduction to the key concepts and how you can contribute effectively.

---

If you're passionate about **renewable energy, open-source collaboration, and sustainability**, please join us in advancing solar forecasting solutions for a better future! 🌍☀️✨

## Contributing and community

[![issues badge](https://img.shields.io/github/issues/openclimatefix/open-data-pvnet?color=FFAC5F)](https://github.com/openclimatefix/open-data-pvnet/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc)

- PR's are welcome! See the [Organisation Profile](https://github.com/openclimatefix) for details on contributing
- Find out about our other projects in the [here](https://github.com/openclimatefix/.github/tree/main/profile)
- Check out the [OCF blog](https://openclimatefix.org/blog) for updates
- Follow OCF on [LinkedIn](https://uk.linkedin.com/company/open-climate-fix)

---

## GitHub Project
Explore our project board to track progress and tasks: [Open Climate Fix Solar Project Board](https://github.com/orgs/openclimatefix/projects/36)

---

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/peterdudfield"><img src="https://avatars.githubusercontent.com/u/34686298?v=4?s=100" width="100px;" alt="Peter Dudfield"/><br /><sub><b>Peter Dudfield</b></sub></a><br /><a href="#projectManagement-peterdudfield" title="Project Management">📆</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Sukh-P"><img src="https://avatars.githubusercontent.com/u/42407101?v=4?s=100" width="100px;" alt="Sukhil Patel"/><br /><sub><b>Sukhil Patel</b></sub></a><br /><a href="#projectManagement-Sukh-P" title="Project Management">📆</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jcamier"><img src="https://avatars.githubusercontent.com/u/14153557?v=4?s=100" width="100px;" alt="Jack Camier"/><br /><sub><b>Jack Camier</b></sub></a><br /><a href="#ideas-jcamier" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!


*Part of the [Open Climate Fix](https://github.com/orgs/openclimatefix/people) community.*

[![OCF Logo](https://cdn.prod.website-files.com/62d92550f6774db58d441cca/6324a2038936ecda71599a8b_OCF_Logo_black_trans.png)](https://openclimatefix.org)

## License

This project is licensed under the [MIT License](LICENSE). See the LICENSE file for details.
