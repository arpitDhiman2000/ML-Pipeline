# AWS deployment guide (Sprint 5)

The CD workflow (`.github/workflows/cd.yml`) builds the image, pushes it to
**ECR**, and deploys to **EC2**. It is **dormant** until you set the
`ENABLE_CD` repo variable, so the repo stays green until you opt in. Everything
below is a one-time setup you do in your own AWS account.

> Cost: ECR storage + a `t2.micro`/`t3.micro` EC2 are free-tier eligible for 12
> months. **Stop/terminate the EC2 instance when not demoing** to avoid charges.

---

## 1. Install + configure the AWS CLI

```powershell
winget install -e --id Amazon.AWSCLI      # or download the MSI from AWS
aws --version
```
Create an **IAM user** (Console → IAM → Users → Create user) with programmatic
access, attach the least-privilege policy in §4, then:
```powershell
aws configure          # paste Access key, Secret, region (e.g. ap-south-1), json
aws sts get-caller-identity   # verify
```

## 2. Create the ECR repository

```powershell
aws ecr create-repository --repository-name threat-detection --region ap-south-1
```

## 3. Create S3 buckets for the data zones (DVC remote + zones)

```powershell
aws s3 mb s3://threat-detection-<your-unique-suffix> --region ap-south-1
```
Point DVC at S3 as a second remote (keep DagsHub too — DVC supports many):
```powershell
uv sync --group cloud
uv run dvc remote add s3remote s3://threat-detection-<suffix>/dvc
uv run dvc push -r s3remote          # data/artifacts now in S3
```
This is the "AWS S3 raw/processed/artifacts zones" wiring.

## 4. Least-privilege IAM policy (attach to the CI/CD user)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Sid": "ECR", "Effect": "Allow",
     "Action": ["ecr:GetAuthorizationToken","ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer","ecr:BatchGetImage",
                "ecr:PutImage","ecr:InitiateLayerUpload","ecr:UploadLayerPart",
                "ecr:CompleteLayerUpload"],
     "Resource": "*"},
    {"Sid": "S3Zones", "Effect": "Allow",
     "Action": ["s3:GetObject","s3:PutObject","s3:ListBucket"],
     "Resource": ["arn:aws:s3:::threat-detection-*",
                  "arn:aws:s3:::threat-detection-*/*"]}
  ]
}
```

## 5. Launch an EC2 instance (deploy target)

* Amazon Linux 2023, `t3.micro`, security group: allow inbound 22 (your IP) + 80.
* Install Docker + AWS CLI on it:
  ```bash
  sudo dnf install -y docker && sudo systemctl enable --now docker
  sudo usermod -aG docker ec2-user
  ```
* Attach an **instance role** with ECR pull permission (or reuse the keys).

## 6. Set GitHub repo variables + secrets

Repo → Settings → Secrets and variables → Actions.

**Variables:**
| Name | Value |
|------|-------|
| `ENABLE_CD` | `true` |
| `AWS_REGION` | `ap-south-1` |
| `ECR_REPOSITORY` | `threat-detection` |

**Secrets:**
| Name | Value |
|------|-------|
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | the IAM user keys |
| `EC2_HOST` | instance public IP/DNS |
| `EC2_USER` | `ec2-user` |
| `EC2_SSH_KEY` | the private key (PEM contents) |
| `DAGSHUB_USER` / `DAGSHUB_TOKEN` | for `dvc pull` of artifacts in CD |

## 7. Trigger it

Push to `main` (CD runs after CI succeeds) or run **Actions → CD → Run workflow**.
Then browse `http://<EC2_PUBLIC_IP>/health`.

---

## Local Docker (no AWS needed)

```powershell
uv run dvc pull            # ensure artifacts/ is present
docker compose up --build  # service on http://localhost:8000
```
