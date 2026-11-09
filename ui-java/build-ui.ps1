param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$AppName = "SelfGrowingTraderControlV193",
    [switch]$JarOnly
)

$ErrorActionPreference = "Stop"

$srcDir = Join-Path $PSScriptRoot "src"
$buildDir = Join-Path $PSScriptRoot "build"
$classesDir = Join-Path $buildDir ("classes-" + $AppName)
$jarPath = Join-Path $buildDir "$AppName.jar"
$packageInputDir = Join-Path $buildDir ("package-input-" + $AppName)
$distDir = Join-Path $PSScriptRoot "dist"
$appImage = Join-Path $distDir $AppName

if (Test-Path $classesDir) {
    Remove-Item -LiteralPath $classesDir -Recurse -Force
}
New-Item -ItemType Directory -Path $classesDir | Out-Null
New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
New-Item -ItemType Directory -Path $distDir -Force | Out-Null

$sources = Get-ChildItem -Path $srcDir -Recurse -Filter *.java | ForEach-Object { $_.FullName }
if (-not $sources) {
    throw "No Java sources found under $srcDir"
}

$sourceList = Join-Path $buildDir ("sources-" + $AppName + ".txt")
$sources | ForEach-Object { '"' + ($_ -replace '\\', '/') + '"' } | Set-Content -Path $sourceList -Encoding ASCII
& javac --release 17 -encoding UTF-8 -d $classesDir "@$sourceList"
if ($LASTEXITCODE -ne 0) {
    throw "javac failed with exit code $LASTEXITCODE"
}

if (Test-Path $jarPath) {
    Remove-Item -LiteralPath $jarPath -Force
}
& jar --create --file $jarPath --main-class com.selfgrowingtrader.control.Main -C $classesDir .
if ($LASTEXITCODE -ne 0) {
    throw "jar failed with exit code $LASTEXITCODE"
}

& java -jar $jarPath --self-test
if ($LASTEXITCODE -ne 0) {
    throw "V193 Five/bridge-v10 UI contract failed with exit code $LASTEXITCODE"
}

if ($JarOnly) {
    if (Test-Path -LiteralPath $appImage) {
        Write-Warning "JarOnly khong cap nhat EXE hien co tai $appImage. Chay build-ui.ps1 khong co -JarOnly de tao lai app image va launcher .cfg."
    }
    Write-Host "Built JAR: $jarPath"
    return
}

if (Test-Path $appImage) {
    try {
        Remove-Item -LiteralPath $appImage -Recurse -Force
    } catch {
        throw "Khong the thay the app image $appImage. Hay dong SelfGrowingTraderControlV193.exe roi chay lai. Chi tiet: $($_.Exception.Message)"
    }
}

# jpackage sao chép toàn bộ --input vào app image. Dùng staging chỉ chứa JAR
# hiện tại để các JAR/classes UI lịch sử không bị đóng gói vào V193.
if (Test-Path -LiteralPath $packageInputDir) {
    Remove-Item -LiteralPath $packageInputDir -Recurse -Force
}
New-Item -ItemType Directory -Path $packageInputDir | Out-Null
Copy-Item -LiteralPath $jarPath -Destination (Join-Path $packageInputDir "$AppName.jar")

$jpackage = Get-Command jpackage -ErrorAction SilentlyContinue
if ($jpackage) {
    & jpackage `
        --type app-image `
        --name $AppName `
        --input $packageInputDir `
        --main-jar "$AppName.jar" `
        --main-class com.selfgrowingtrader.control.Main `
        --dest $distDir `
        --arguments "`"$ProjectRoot`""
    if ($LASTEXITCODE -ne 0) {
        throw "jpackage failed with exit code $LASTEXITCODE"
    }
    $packagedJar = Join-Path (Join-Path $appImage "app") "$AppName.jar"
    $launcherConfig = Join-Path (Join-Path $appImage "app") "$AppName.cfg"
    $packageManifest = Join-Path (Join-Path $appImage "app") ".jpackage.xml"
    foreach ($requiredArtifact in @($packagedJar, $launcherConfig, $packageManifest)) {
        if (-not (Test-Path -LiteralPath $requiredArtifact -PathType Leaf)) {
            throw "jpackage tao app image khong day du; thieu artifact bat buoc: $requiredArtifact"
        }
    }
    Write-Host "Verified launcher config: $launcherConfig"
    Write-Host "Built EXE: $(Join-Path $appImage "$AppName.exe")"
} else {
    Write-Warning "jpackage not found; built jar only."
}

if (Test-Path -LiteralPath $packageInputDir) {
    Remove-Item -LiteralPath $packageInputDir -Recurse -Force
}

# V193 thay V191 bằng profile tìm edge theo bằng chứng. Chỉ
# dọn đúng artifact đã bị
# thay thế sau khi JAR mới compile và vượt command-contract self-test.
foreach ($legacyName in @("SelfGrowingTraderControlV191", "SelfGrowingTraderControlV190", "SelfGrowingTraderControlV188", "SelfGrowingTraderControlV11")) {
    if ($legacyName -eq $AppName) { continue }
    foreach ($legacyPath in @(
        (Join-Path $buildDir "$legacyName.jar"),
        (Join-Path $buildDir "${legacyName}_Patched.jar"),
        (Join-Path $buildDir "classes-$legacyName"),
        (Join-Path $buildDir "sources-$legacyName.txt"),
        (Join-Path $distDir $legacyName)
    )) {
        if (Test-Path -LiteralPath $legacyPath) {
            try {
                Remove-Item -LiteralPath $legacyPath -Recurse -Force
            } catch {
                Write-Warning "Deferred cleanup for locked replaced UI artifact: $legacyPath"
            }
        }
    }
}

# Các build V6..V10.2 đã được thay thế và từng bị jpackage cũ kéo nhầm vào
# app image. Chỉ dọn artifact biên dịch trong ui-java/build; source V193 không
# bị ảnh hưởng.
$legacyCompiledArtifacts = @(
    Get-ChildItem -LiteralPath $buildDir -File | Where-Object {
        ($_.Name -like "SelfGrowingTraderControl*.jar" -and $_.Name -ne "$AppName.jar") -or
        ($_.Name -like "sources*.txt" -and $_.Name -ne "sources-$AppName.txt")
    }
    Get-ChildItem -LiteralPath $buildDir -Directory | Where-Object {
        ($_.Name -eq "classes" -or $_.Name -like "classes-*") -and $_.Name -ne ("classes-" + $AppName)
    }
)
foreach ($legacyPath in $legacyCompiledArtifacts) {
    Remove-Item -LiteralPath $legacyPath.FullName -Recurse -Force
}

# Dọn thư mục compile kiểm tra thủ công từng dùng khi phát triển V190. Đây không
# phải output phân phối và không được để lẫn với classes của build chính.
$temporaryCompileCheck = Join-Path $buildDir "classes-v190-check2"
if (Test-Path -LiteralPath $temporaryCompileCheck) {
    Remove-Item -LiteralPath $temporaryCompileCheck -Recurse -Force
}

Write-Host "Built JAR: $jarPath"
