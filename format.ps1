# 定义目标文件夹路径
$folderPath = "test"

# 定义要格式化的文件扩展名
$extensions = @("*.cpp", "*.h", "*.hpp", "*.c")

# 遍历文件夹中的所有文件
foreach ($extension in $extensions)
{
    $files = Get-ChildItem -Path $folderPath -Recurse -Filter $extension
    foreach ($file in $files)
    {
        # 获取文件的完整路径
        $filePath = $file.FullName
        Write-Host "Formatting file: $filePath"

        # 使用 clang-format 格式化文件
        clang-format -i $filePath
    }
}

Write-Host "Formatting completed."