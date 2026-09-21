# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

RUN python -m pip install --no-cache-dir mitmproxy==11.1.3

WORKDIR /opt/closed-book-proxy
COPY allowlist-base.txt entrypoint.sh rewriter.py ./
RUN chmod +x ./entrypoint.sh

EXPOSE 3128 3129
ENTRYPOINT ["./entrypoint.sh"]
