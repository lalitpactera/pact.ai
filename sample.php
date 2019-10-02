<?php
echo('alert');
$data = $_POST['jsonString'];
//set mode of file to writable.
chmod("data.json",0777);
$f = fopen("data.json", "w+") or die("fopen failed");
fwrite($f, $data);
fclose($f);
#$fn = fopen("filename.txt","r");
#$result = fgets($fn);
#fclose($fn);
$str1 = 'python backend_workbench.py ';
$str2 = shell_exec($str1);
?>
