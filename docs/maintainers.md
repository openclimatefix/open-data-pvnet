# Maintainers Guide

This document provides step-by-step instructions for maintainers of the **open-data-pvnet** PyPI project.

---

## Prerequisites

Ensure the following before starting:

1. **Python**: Version >= 3.9 installed.
2. **Git**: Ensure Git is installed on your system.

---

## Setting Up the Project Locally

### 1. Clone the Repository

Clone the repository from GitHub:
```bash
git clone git@github.com:openclimatefix/open-data-pvnet.git
cd open-data-pvnet
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

### 3. Install the dependencies, including development dependencies:

```bash
pip install -e .
pip install -e ".[dev]"
```

- run basic verifications (It should display the version number in the console):

```bash
python -c "import open_data_pvnet; print(open_data_pvnet.__version__)"
```

### 4. Set up Environment Variables

Look for the .env.local file in the root directory and set the environment variables as needed.

**For AWS Credentials:**

- Click on the link to create an AWS account: [AWS Console](https://aws.amazon.com/console/).

- Click on the user icon in the top right corner and select "My Security Credentials".

- Click on "Access keys (access key ID and secret access key)".

- Click on "Create New Access Key".

- Copy the "Access Key ID" and "Secret Access Key" and paste them in the .env.local file.

**For Hugging Face Credentials:**

- Click on the link to create a Hugging Face account: [Hugging Face](https://huggingface.co/).

- Click on the user icon in the top right corner and select "API token".

- Copy the API token and paste it in the .env.local file.



### 5. Code Quality and Testing

For detailed instructions on maintaining code quality and running tests, refer to the [Development and Testing Guide](getting_started.md#development-and-testing-guide) in our getting started documentation. This guide covers:

- Using `ruff` for linting
- Formatting with `black`
- Running tests with `pytest`


### 6. Eventually this repo will be on pypi.org ...stay tuned

