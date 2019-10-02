<?php
echo('alert');
$data = $_POST['jsonString'];

$arr = explode('@', $data);
$filename = $arr[0];
$filename = substr($filename, 1, strlen($data)-2);

$data = $arr[1];
$data = substr($data, 0, strlen($data)-1);
$data = stripslashes($data);

#$f = fopen("filename1.txt", "w+") or die("fopen failed");
#fwrite($f, $data);
#fclose($f);

$json_data = json_decode($data,true);
file_put_contents($filename, json_encode($json_data));
?>