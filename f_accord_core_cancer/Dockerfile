FROM mcr.microsoft.com/dotnet/core/runtime:3.1 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/core/sdk:3.1 AS build
WORKDIR /src
COPY ["f_accord_core_cancer/f_accord_core_cancer.fsproj", "f_accord_core_cancer/"]
RUN dotnet restore "f_accord_core_cancer/f_accord_core_cancer.fsproj"
COPY . .
WORKDIR "/src/f_accord_core_cancer"
RUN dotnet build "f_accord_core_cancer.fsproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "f_accord_core_cancer.fsproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "f_accord_core_cancer.dll"]
