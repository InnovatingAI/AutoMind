FROM automind-base

# where to put submission.csv, will be extracted
ARG SUBMISSION_DIR
ENV SUBMISSION_DIR=${SUBMISSION_DIR}
# where to put any logs, will be extracted
ARG LOGS_DIR
ENV LOGS_DIR=${LOGS_DIR}
# where to put any code, will be extracted
ARG CODE_DIR
ENV CODE_DIR=${CODE_DIR}
# where to put any other agent-specific files, will not be necessarily extracted
ARG AGENT_DIR
ENV AGENT_DIR=${AGENT_DIR}

# put all the agent files in the expected location
RUN mkdir ${AGENT_DIR}/automind
COPY ./automind ${AGENT_DIR}/automind
COPY ./mle-bench/start.sh ${AGENT_DIR}/start.sh
COPY ./mle-bench/additional_notes.txt ${AGENT_DIR}/additional_notes.txt