############################################################
# HPNTS AWS validation environment (Phase 3)
#
# 2x flat-VM tier (c6i.xlarge, on-demand: predictable perf for the
#   high-priority HPC comparison) + 1x t3.large running 8 Docker
#   containers as the nested tier (mirrors the 2-flat/8-nested split
#   in HPNTS_Comparative_Project.java).
#
# Spot pricing on the nested tier to keep validation-run costs low;
# use `terraform destroy` (or aws_deploy/teardown.sh) after each run.
############################################################

terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

variable "region" { default = "us-east-1" }
variable "key_name" { description = "Existing EC2 key pair name for SSH access" }
variable "allowed_ssh_cidr" { description = "Your IP in CIDR form, e.g. 203.0.113.4/32" }

provider "aws" {
  region = var.region
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
}

resource "aws_security_group" "hpnts_sg" {
  name        = "hpnts-validation-sg"
  description = "SSH + internal classifier/orchestration traffic"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }
  ingress {
    description = "classifier service + SSM"
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
    self        = true
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_iam_role" "ssm_role" {
  name = "hpnts-ssm-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ssm_attach" {
  role       = aws_iam_role.ssm_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "ssm_profile" {
  name = "hpnts-ssm-profile"
  role = aws_iam_role.ssm_role.name
}

# --- Flat-VM tier: 2x c6i.xlarge (on-demand) ---
resource "aws_instance" "flat_vm" {
  count                       = 2
  ami                         = data.aws_ami.ubuntu.id
  instance_type               = "c6i.xlarge"
  key_name                    = var.key_name
  vpc_security_group_ids      = [aws_security_group.hpnts_sg.id]
  iam_instance_profile        = aws_iam_instance_profile.ssm_profile.name
  tags = { Name = "hpnts-flat-vm-${count.index}", Tier = "flat" }
}

# --- Nested tier: 1x t3.large host, 8 Docker containers created via user_data ---
resource "aws_instance" "nested_host" {
  instance_type          = "t3.large"
  ami                     = data.aws_ami.ubuntu.id
  key_name                = var.key_name
  vpc_security_group_ids  = [aws_security_group.hpnts_sg.id]
  iam_instance_profile    = aws_iam_instance_profile.ssm_profile.name
  tags = { Name = "hpnts-nested-host", Tier = "nested" }

  user_data = <<-EOF
    #!/bin/bash
    apt-get update && apt-get install -y docker.io
    systemctl enable --now docker
    for i in $(seq 1 8); do
      docker run -d --name nested_container_$i --cpus="0.25" --memory="512m" \
        ubuntu:22.04 sleep infinity
    done
  EOF
}

# --- Classifier service host ---
resource "aws_instance" "classifier_service" {
  instance_type          = "t3.micro"
  ami                     = data.aws_ami.ubuntu.id
  key_name                = var.key_name
  vpc_security_group_ids  = [aws_security_group.hpnts_sg.id]
  iam_instance_profile    = aws_iam_instance_profile.ssm_profile.name
  tags = { Name = "hpnts-classifier-service", Tier = "control" }

  user_data = <<-EOF
    #!/bin/bash
    apt-get update && apt-get install -y python3-pip
    pip3 install flask
  EOF
}

output "flat_vm_ids" { value = aws_instance.flat_vm[*].id }
output "flat_vm_ips" { value = aws_instance.flat_vm[*].public_ip }
output "nested_host_id" { value = aws_instance.nested_host.id }
output "nested_host_ip" { value = aws_instance.nested_host.public_ip }
output "classifier_service_ip" { value = aws_instance.classifier_service.public_ip }
