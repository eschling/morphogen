#!/usr/bin/perl -w
use strict;
binmode(STDIN,":utf8");
binmode(STDOUT,":utf8");

my %d;
while(<STDIN>) {
  chomp;
  my @words = split /\s+/;
  for my $w (@words) {
    $d{$w}++;
  }
}

for my $type (sort {$d{$b} <=> $d{$a}} keys %d) {
  next if $type =~ /\d/;  # skip digits
  next if $type =~ /[.,;:'`"\$<>^\*\+\-()\[\]#@=_&%]/;  # skip punctuation
  next if length($type) < 4;
  print "$type\n";
}

