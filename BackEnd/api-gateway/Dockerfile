# Stage 1: Build
FROM maven:3.8-eclipse-temurin-17-alpine AS build
RUN apk upgrade --no-cache
WORKDIR /build

# Build common-library
COPY common-library/pom.xml ./common-library/pom.xml
COPY common-library/src ./common-library/src
RUN cd common-library && mvn clean install -DskipTests -B

# Build api-gateway
WORKDIR /build/service
COPY api-gateway/pom.xml .
RUN mvn dependency:go-offline -B
COPY api-gateway/src ./src
RUN mvn clean package -DskipTests -B

# Stage 2: Run
FROM eclipse-temurin:17-jre-alpine
RUN apk upgrade --no-cache
WORKDIR /app
COPY --from=build /build/service/target/*.jar app.jar

ARG SPRING_PROFILES_ACTIVE=prod
ENV SPRING_PROFILES_ACTIVE=${SPRING_PROFILES_ACTIVE}

EXPOSE 8080
ENTRYPOINT ["java", \
    "-XX:+UseContainerSupport", \
    "-XX:MaxRAMPercentage=75.0", \
    "-Djava.security.egd=file:/dev/./urandom", \
    "-jar", "app.jar"]